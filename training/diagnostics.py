"""Shared numerical helpers for prefix-channel diagnostics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats


def gate_shape(layers) -> tuple[int, int, int]:
    if not layers:
        raise ValueError("Model has no KV-prefix layers")
    shapes = [tuple(layer._ablate_mask.shape) for layer in layers]
    if any(len(shape) != 2 for shape in shapes) or len(set(shapes)) != 1:
        raise ValueError(f"Expected one equal [head, prefix] mask per layer, got {shapes}")
    return len(layers), shapes[0][0], shapes[0][1]


def effective_gate_grid(layers) -> torch.Tensor:
    """Return signed effective coefficients as a CPU [layer, head, prefix] tensor."""
    gate_shape(layers)
    coefficients = []
    for layer in layers:
        if hasattr(layer, "weight"):
            effective = torch.tanh(layer.weight)
        elif hasattr(layer, "scale") and hasattr(layer, "gate"):
            effective = torch.sigmoid(layer.gate) * layer.scale
        elif hasattr(layer, "gate"):
            effective = torch.tanh(layer.gate)
        else:
            raise TypeError(f"Cannot recover an effective gate from {type(layer).__name__}")
        if effective.shape != layer._ablate_mask.shape:
            raise ValueError(
                f"Gate shape {tuple(effective.shape)} does not match mask "
                f"{tuple(layer._ablate_mask.shape)}"
            )
        coefficients.append(effective.detach().float().cpu())
    return torch.stack(coefficients)


@contextmanager
def ablate(layers, cells=None):
    """Temporarily zero selected cells, or the full prefix when ``cells`` is None."""
    shape = gate_shape(layers)
    previous = [layer._ablate_mask.detach().clone() for layer in layers]
    try:
        if cells is None:
            for layer in layers:
                layer._ablate_mask.zero_()
        else:
            for layer_idx, head_idx, prefix_idx in cells:
                if not (
                    0 <= layer_idx < shape[0]
                    and 0 <= head_idx < shape[1]
                    and 0 <= prefix_idx < shape[2]
                ):
                    raise IndexError((layer_idx, head_idx, prefix_idx))
                layers[layer_idx]._ablate_mask[head_idx, prefix_idx] = 0
        yield
    finally:
        for layer, mask in zip(layers, previous, strict=True):
            layer._ablate_mask.copy_(mask)


def joint_sign_flip_pvalues(
    deltas,
    n_draws: int,
    seed: int,
    *,
    device: str | torch.device = "cpu",
    chunk_size: int = 1_000,
) -> np.ndarray:
    """Two-sided Monte Carlo p-values using shared signs for all hypotheses."""
    if n_draws <= 0 or chunk_size <= 0:
        raise ValueError("n_draws and chunk_size must be positive")
    target = torch.device(device)
    values = torch.as_tensor(deltas, dtype=torch.float32, device=target)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("deltas must have shape [hypothesis, paired observation]")
    observed = values.mean(dim=1).abs()
    exceedances = torch.zeros(values.shape[0], dtype=torch.int64, device=target)
    generator = torch.Generator(device=target).manual_seed(seed)
    for start in range(0, n_draws, chunk_size):
        count = min(chunk_size, n_draws - start)
        signs = torch.randint(
            0,
            2,
            (count, values.shape[1]),
            generator=generator,
            device=target,
            dtype=torch.int8,
        ).to(torch.float32)
        signs.mul_(2).sub_(1)
        randomized = (signs @ values.T).abs().div_(values.shape[1])
        exceedances += (randomized >= observed.unsqueeze(0) - 1e-12).sum(dim=0)
    return ((exceedances + 1).double() / (n_draws + 1)).cpu().numpy()


def paired_rank_stats(
    reference_ranks,
    intervention_ranks,
    *,
    n_boot: int,
    n_draws: int,
    seed: int,
    sign_flip_device: str | torch.device = "cpu",
    chunk_size: int = 1_000,
) -> dict[str, float]:
    reference = np.asarray(reference_ranks, dtype=np.float64)
    intervention = np.asarray(intervention_ranks, dtype=np.float64)
    if reference.shape != intervention.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError("Paired rank vectors must have the same non-empty shape")
    if np.any(reference < 1) or np.any(intervention < 1) or n_boot <= 0:
        raise ValueError("Ranks and n_boot must be positive")
    deltas = 1.0 / intervention - 1.0 / reference
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, chunk_size):
        count = min(chunk_size, n_boot - start)
        indices = rng.integers(0, deltas.size, size=(count, deltas.size))
        bootstrap[start : start + count] = deltas[indices].mean(axis=1)
    lo, hi = np.percentile(bootstrap, [2.5, 97.5])
    pvalue = joint_sign_flip_pvalues(
        deltas[None, :],
        n_draws,
        seed,
        device=sign_flip_device,
        chunk_size=chunk_size,
    )[0]
    return {
        "delta_mrr": float(deltas.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "sign_flip_p": float(pvalue),
    }


def benjamini_hochberg(pvalues) -> np.ndarray:
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pvalues must be a non-empty finite vector")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    adjusted = values[order] * values.size / np.arange(1, values.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def magnitude_correlations(effective_gate, delta_mrr) -> dict[str, dict[str, float | None]]:
    magnitude = np.abs(np.asarray(effective_gate, dtype=np.float64).reshape(-1))
    delta = np.asarray(delta_mrr, dtype=np.float64).reshape(-1)
    if magnitude.shape != delta.shape or magnitude.size < 2:
        raise ValueError("effective_gate and delta_mrr must contain the same cells")

    def correlation(outcome):
        if np.ptp(magnitude) == 0 or np.ptp(outcome) == 0:
            return {"rho": None, "pvalue": None}
        result = stats.spearmanr(magnitude, outcome)
        rho, pvalue = float(result.statistic), float(result.pvalue)
        return {
            "rho": rho if np.isfinite(rho) else None,
            "pvalue": pvalue if np.isfinite(pvalue) else None,
        }

    return {
        "gate_magnitude_vs_delta_mrr": correlation(delta),
        "gate_magnitude_vs_mrr_loss": correlation(-delta),
    }


def _random_derangement(size: int, generator: torch.Generator) -> torch.Tensor:
    identity = torch.arange(size)
    while True:
        permutation = torch.randperm(size, generator=generator)
        if not bool((permutation == identity).any()):
            return permutation


def deranged_permutation(batch_sizes, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    parts = []
    singleton_indices = []
    offset = 0
    for size in batch_sizes:
        size = int(size)
        if size <= 0:
            raise ValueError("Batch sizes must be positive")
        if size == 1:
            parts.append(torch.tensor([offset]))
            singleton_indices.append(offset)
            offset += 1
            continue
        permutation = _random_derangement(size, generator)
        parts.append(permutation + offset)
        offset += size
    if not parts:
        raise ValueError("At least one batch is required")
    permutation = torch.cat(parts)
    if len(singleton_indices) > 1:
        singleton_targets = _random_derangement(len(singleton_indices), generator)
        singleton_indices = torch.tensor(singleton_indices)
        permutation[singleton_indices] = singleton_indices[singleton_targets]
    elif singleton_indices:
        singleton = singleton_indices[0]
        candidates = torch.arange(permutation.numel())
        candidates = candidates[candidates != singleton]
        if candidates.numel() == 0:
            raise ValueError("At least two rows are required for derangement")
        donor = candidates[torch.randint(candidates.numel(), (), generator=generator)]
        permutation[singleton], permutation[donor] = (
            permutation[donor].clone(),
            permutation[singleton].clone(),
        )
    return permutation


def prefix_cosines(
    prefixes_a: torch.Tensor,
    prefixes_b: torch.Tensor,
    batch_sizes,
    *,
    negative_seed: int,
) -> dict[str, np.ndarray]:
    """Matched and deranged cosine, raw and side-batch-mean-centred."""
    a = prefixes_a.detach().float().cpu()
    b = prefixes_b.detach().float().cpu()
    sizes = [int(size) for size in batch_sizes]
    if a.shape != b.shape or a.ndim != 3 or sum(sizes) != a.shape[0]:
        raise ValueError("Expected equal [N, prefix, dimension] tensors matching batch_sizes")
    centered_a = _center_by_batch(a, sizes)
    centered_b = _center_by_batch(b, sizes)
    permutation = deranged_permutation(sizes, negative_seed)

    def cosine(left, right):
        return F.cosine_similarity(left.flatten(1), right.flatten(1), dim=1).numpy()

    return {
        "raw_matched": cosine(a, b),
        "raw_deranged": cosine(a, b[permutation]),
        "centered_matched": cosine(centered_a, centered_b),
        "centered_deranged": cosine(centered_a, centered_b[permutation]),
        "permutation": permutation.numpy(),
    }


def _center_by_batch(prefixes: torch.Tensor, batch_sizes: list[int]) -> torch.Tensor:
    centered = torch.empty_like(prefixes)
    ranges = []
    start = 0
    for size in batch_sizes:
        stop = start + size
        ranges.append((start, stop))
        if size > 1:
            chunk = prefixes[start:stop]
            centered[start:stop] = chunk - chunk.mean(dim=0, keepdim=True)
        start = stop

    singletons = [start for start, stop in ranges if stop - start == 1]
    if len(singletons) > 1:
        values = prefixes[singletons]
        centered[singletons] = values - values.mean(dim=0, keepdim=True)
    elif singletons:
        reference = next(
            (prefixes[start:stop] for start, stop in reversed(ranges) if stop - start > 1),
            None,
        )
        if reference is None:
            raise ValueError("At least two rows are required for batch centering")
        centered[singletons[0]] = prefixes[singletons[0]] - reference.mean(dim=0)
    return centered


def summarize(values) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("values must be a non-empty vector")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def save_npy_atomic(path: str | Path, array) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(array), allow_pickle=False)
    temporary.replace(destination)
