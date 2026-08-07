"""Batch reduced Siamese examples for baseline and GNN models."""

import math
from numbers import Real
from typing import Any

import torch

PAD_TOKEN_ID = 1
MASK_TOKEN_ID = 5

NUM_BASE_EDGE_TYPES = 5
REVERSE_TYPE_OFFSET = NUM_BASE_EDGE_TYPES
SELF_LOOP_TYPE = 2 * NUM_BASE_EDGE_TYPES


def _mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_prob: float = 0.15,
    mask_token_id: int = MASK_TOKEN_ID,
    ignore_index: int = -100,
) -> tuple:
    """Replace sampled real tokens with the CLAP mask token."""
    masked_ids = input_ids.clone()
    mlm_labels = torch.full_like(input_ids, ignore_index)

    real_mask = attention_mask.bool()
    rand = torch.rand_like(input_ids, dtype=torch.float)
    mask_positions = real_mask & (rand < mask_prob)

    for i in range(input_ids.size(0)):
        seq_real = real_mask[i]
        if seq_real.any() and not mask_positions[i].any():
            real_indices = seq_real.nonzero(as_tuple=True)[0]
            pick = real_indices[torch.randint(len(real_indices), (1,))]
            mask_positions[i, pick] = True

    mlm_labels[mask_positions] = input_ids[mask_positions]
    masked_ids[mask_positions] = mask_token_id

    return masked_ids, mlm_labels


def collate_baseline(batch: list[dict]) -> dict[str, Any]:
    """Pad a Siamese batch and add masked-token training targets."""
    binary_1_list = [s["binary_1"] for s in batch]
    binary_2_list = [s["binary_2"] for s in batch]
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    metadata = [s["metadata"] for s in batch]

    b1 = _collate_binaries(binary_1_list)
    b2 = _collate_binaries(binary_2_list)

    masked_ids_a, mlm_labels_a = _mask_tokens(b1["input_ids"], b1["attention_mask"])
    masked_ids_b, mlm_labels_b = _mask_tokens(b2["input_ids"], b2["attention_mask"])

    return {
        "input_ids_a": b1["input_ids"],
        "attention_mask_a": b1["attention_mask"],
        "token_type_ids_a": b1["token_type_ids"],
        "function_features_a": b1["function_features"],
        "masked_input_ids_a": masked_ids_a,
        "mlm_labels_a": mlm_labels_a,
        "input_ids_b": b2["input_ids"],
        "attention_mask_b": b2["attention_mask"],
        "token_type_ids_b": b2["token_type_ids"],
        "function_features_b": b2["function_features"],
        "masked_input_ids_b": masked_ids_b,
        "mlm_labels_b": mlm_labels_b,
        "labels": labels,
        "metadata": metadata,
    }


def collate_baseline_inference(batch: list[dict]) -> dict[str, Any]:
    """Pad a Siamese batch without random token masking."""
    binary_1_list = [s["binary_1"] for s in batch]
    binary_2_list = [s["binary_2"] for s in batch]
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    metadata = [s["metadata"] for s in batch]

    b1 = _collate_binaries(binary_1_list)
    b2 = _collate_binaries(binary_2_list)

    return {
        "input_ids_a": b1["input_ids"],
        "attention_mask_a": b1["attention_mask"],
        "token_type_ids_a": b1["token_type_ids"],
        "function_features_a": b1["function_features"],
        "input_ids_b": b2["input_ids"],
        "attention_mask_b": b2["attention_mask"],
        "token_type_ids_b": b2["token_type_ids"],
        "function_features_b": b2["function_features"],
        "labels": labels,
        "metadata": metadata,
    }


def collate_gnn(
    batch: list[dict],
    use_edge_features: bool = False,
    use_reverse_edges: bool = False,
) -> dict[str, Any]:
    """Pad tokens and assemble PyG-style graph tensors for both pair sides."""
    binary_1_list = [s["binary_1"] for s in batch]
    binary_2_list = [s["binary_2"] for s in batch]
    labels = torch.tensor([s["label"] for s in batch], dtype=torch.long)
    metadata = [s["metadata"] for s in batch]

    b1 = _collate_binaries(binary_1_list)
    b2 = _collate_binaries(binary_2_list)
    g1 = _collate_graphs(binary_1_list, use_edge_features, use_reverse_edges)
    g2 = _collate_graphs(binary_2_list, use_edge_features, use_reverse_edges)

    out = {
        "input_ids_a": b1["input_ids"],
        "attention_mask_a": b1["attention_mask"],
        "token_type_ids_a": b1["token_type_ids"],
        "function_features_a": b1["function_features"],
        "node_features_a": g1["node_features"],
        "edge_index_a": g1["edge_index"],
        "edge_types_a": g1["edge_types"],
        "graph_batch_a": g1["batch"],
        "input_ids_b": b2["input_ids"],
        "attention_mask_b": b2["attention_mask"],
        "token_type_ids_b": b2["token_type_ids"],
        "function_features_b": b2["function_features"],
        "node_features_b": g2["node_features"],
        "edge_index_b": g2["edge_index"],
        "edge_types_b": g2["edge_types"],
        "graph_batch_b": g2["batch"],
        "labels": labels,
        "metadata": metadata,
    }

    if use_edge_features:
        out["gnn_edge_index_a"] = g1["gnn_edge_index"]
        out["gnn_edge_types_a"] = g1["gnn_edge_types"]
        out["gnn_edge_index_b"] = g2["gnn_edge_index"]
        out["gnn_edge_types_b"] = g2["gnn_edge_types"]

    return out


def collate_functions(
    functions: list[dict],
    *,
    include_function_features: bool = True,
    include_graph: bool = False,
    use_edge_features: bool = False,
    use_reverse_edges: bool = False,
) -> dict[str, Any]:
    """Collate independent functions without creating Siamese pair tensors."""
    out = _collate_binaries(
        functions,
        include_function_features=include_function_features,
    )
    if include_graph:
        out.update(_collate_graphs(functions, use_edge_features, use_reverse_edges))
    return out


def _rand_shape(
    shape,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Generate random values while respecting the generator's device."""
    if generator is None:
        return torch.rand(shape, device=device)
    gen_device = generator.device
    if gen_device == device:
        return torch.rand(shape, generator=generator, device=device)
    x = torch.rand(shape, generator=generator, device=gen_device)
    return x.to(device)


def _rand_index(
    high: int,
    device: torch.device,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if generator is None:
        return torch.randint(high, (1,), device=device)
    index = torch.randint(
        high,
        (1,),
        generator=generator,
        device=generator.device,
    )
    return index.to(device)


def apply_modality_dropout(
    batch: dict[str, Any],
    ratio: float,
    side: str = "a_only",
    pad_token_id: int = PAD_TOKEN_ID,
    generator: torch.Generator | None = None,
    drop_function_features: bool = True,
) -> dict[str, Any]:
    """Drop tokens in place while preserving one real token per nonempty row."""
    if side not in {"a_only", "b_only", "both", "random"}:
        raise ValueError(f"unknown side: {side!r}")
    if not isinstance(ratio, Real) or not math.isfinite(ratio) or not 0.0 <= ratio < 1.0:
        raise ValueError(f"modality dropout ratio must be in [0, 1), got {ratio}")
    if ratio == 0.0:
        return batch

    if side == "a_only":
        _dropout_one_side(batch, "a", ratio, pad_token_id, generator, drop_function_features)
    elif side == "b_only":
        _dropout_one_side(batch, "b", ratio, pad_token_id, generator, drop_function_features)
    elif side == "both":
        _dropout_one_side(batch, "a", ratio, pad_token_id, generator, drop_function_features)
        _dropout_one_side(batch, "b", ratio, pad_token_id, generator, drop_function_features)
    else:  # "random"
        input_ids_a = batch["input_ids_a"]
        batch_size = input_ids_a.size(0)
        # Coin flip lives on the same device as the inputs
        flips = _rand_shape((batch_size,), input_ids_a.device, generator) < 0.5
        _dropout_masked_rows(
            batch, "a", ratio, flips, pad_token_id, generator, drop_function_features
        )
        _dropout_masked_rows(
            batch, "b", ratio, ~flips, pad_token_id, generator, drop_function_features
        )

    return batch


def _dropout_one_side(
    batch: dict[str, Any],
    side: str,
    ratio: float,
    pad_token_id: int,
    generator: torch.Generator | None,
    drop_function_features: bool,
) -> None:
    input_ids = batch[f"input_ids_{side}"]
    attention_mask = batch[f"attention_mask_{side}"]
    device = input_ids.device

    real = attention_mask.bool()
    rand = _rand_shape(input_ids.shape, device, generator)
    drop = real & (rand < ratio)

    # Preserve at least 1 real token per originally-nonempty row
    row_all_dropped = drop.sum(dim=1) == real.sum(dim=1)
    row_has_real = real.any(dim=1)
    rescue_rows = row_all_dropped & row_has_real
    for i in rescue_rows.nonzero(as_tuple=True)[0].tolist():
        real_idx = real[i].nonzero(as_tuple=True)[0]
        keep = real_idx[_rand_index(len(real_idx), device, generator)]
        drop[i, keep] = False

    input_ids[drop] = pad_token_id
    attention_mask[drop] = 0

    if drop_function_features:
        feats = batch.get(f"function_features_{side}")
        if feats is not None:
            # Zero features for every row that actually had tokens dropped.
            # Keep features on rows that were untouched (row_has_real=False
            # rows are no-ops anyway).
            touched = drop.any(dim=1)
            feats[touched] = 0.0


def _dropout_masked_rows(
    batch: dict[str, Any],
    side: str,
    ratio: float,
    row_mask: torch.Tensor,
    pad_token_id: int,
    generator: torch.Generator | None,
    drop_function_features: bool,
) -> None:
    """Drop tokens only on rows where ``row_mask[i]`` is True."""
    if not row_mask.any():
        return
    input_ids = batch[f"input_ids_{side}"]
    attention_mask = batch[f"attention_mask_{side}"]
    device = input_ids.device

    # Make sure row_mask lives on the same device as the tensors it will be
    # broadcast against.
    if row_mask.device != device:
        row_mask = row_mask.to(device)

    real = attention_mask.bool()
    rand = _rand_shape(input_ids.shape, device, generator)
    drop = real & (rand < ratio) & row_mask.unsqueeze(1)

    row_all_dropped = drop.sum(dim=1) == real.sum(dim=1)
    row_has_real = real.any(dim=1)
    rescue_rows = row_all_dropped & row_has_real & row_mask
    for i in rescue_rows.nonzero(as_tuple=True)[0].tolist():
        real_idx = real[i].nonzero(as_tuple=True)[0]
        keep = real_idx[_rand_index(len(real_idx), device, generator)]
        drop[i, keep] = False

    input_ids[drop] = pad_token_id
    attention_mask[drop] = 0

    if drop_function_features:
        feats = batch.get(f"function_features_{side}")
        if feats is not None:
            touched = drop.any(dim=1)
            feats[touched] = 0.0


def _collate_graphs(
    binary_list: list[dict],
    use_edge_features: bool = False,
    use_reverse_edges: bool = False,
) -> dict[str, torch.Tensor]:
    """Assemble forward CFG tensors and optional typed GAT edges."""
    if not binary_list:
        raise ValueError("cannot collate an empty binary list")

    all_node_features = []
    all_edges_src = []
    all_edges_tgt = []
    all_edge_types = []
    all_batch = []

    gnn_src = []
    gnn_tgt = []
    gnn_types = []

    offset = 0

    for graph_idx, b in enumerate(binary_list):
        nf = b.get("node_features", None)
        if isinstance(nf, torch.Tensor):
            if nf.ndim != 2 or nf.shape[0] == 0:
                raise ValueError(
                    f"node_features must have shape [nodes, width], got {tuple(nf.shape)}"
                )
            num_nodes = int(nf.shape[0])
            nf_t = nf.to(torch.float32)
        else:
            if nf is None or len(nf) == 0:
                nf = [[0.0] * 20]
            num_nodes = len(nf)
            nf_t = torch.tensor(nf, dtype=torch.float32)
            if nf_t.ndim != 2:
                raise ValueError(
                    f"node_features must have shape [nodes, width], got {tuple(nf_t.shape)}"
                )

        all_node_features.append(nf_t)
        all_batch.extend([graph_idx] * num_nodes)

        edges = b.get("edges", [])
        raw_etypes = b.get("edge_types", [])
        etypes = [raw_etypes[i] if i < len(raw_etypes) else 2 for i in range(len(edges))]

        for (src, tgt), etype in zip(edges, etypes, strict=False):
            all_edges_src.append(src + offset)
            all_edges_tgt.append(tgt + offset)
            all_edge_types.append(etype)
            if use_edge_features:
                gnn_src.append(src + offset)
                gnn_tgt.append(tgt + offset)
                gnn_types.append(etype)

        if use_edge_features and use_reverse_edges:
            for (src, tgt), etype in zip(edges, etypes, strict=False):
                if src == tgt:
                    continue
                gnn_src.append(tgt + offset)
                gnn_tgt.append(src + offset)
                gnn_types.append(etype + REVERSE_TYPE_OFFSET)

        if use_edge_features:
            nodes_with_orig_self = {src for (src, tgt) in edges if src == tgt}
            for n in range(num_nodes):
                if n in nodes_with_orig_self:
                    continue
                gnn_src.append(n + offset)
                gnn_tgt.append(n + offset)
                gnn_types.append(SELF_LOOP_TYPE)

        offset += num_nodes

    if all_node_features:
        node_features = torch.cat(all_node_features, dim=0)
    else:
        node_features = torch.zeros((0, 20), dtype=torch.float32)

    if all_edges_src:
        edge_index = torch.tensor([all_edges_src, all_edges_tgt], dtype=torch.long)
        edge_types = torch.tensor(all_edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_types = torch.zeros(0, dtype=torch.long)

    batch = torch.tensor(all_batch, dtype=torch.long)

    out = {
        "node_features": node_features,
        "edge_index": edge_index,
        "edge_types": edge_types,
        "batch": batch,
    }

    if use_edge_features:
        if gnn_src:
            gnn_edge_index = torch.tensor([gnn_src, gnn_tgt], dtype=torch.long)
            gnn_edge_types = torch.tensor(gnn_types, dtype=torch.long)
        else:
            gnn_edge_index = torch.zeros(2, 0, dtype=torch.long)
            gnn_edge_types = torch.zeros(0, dtype=torch.long)
        assert gnn_edge_index.size(1) == gnn_edge_types.size(0), (
            f"gnn_edge_index/gnn_edge_types misaligned: "
            f"{gnn_edge_index.size(1)} vs {gnn_edge_types.size(0)}"
        )
        out["gnn_edge_index"] = gnn_edge_index
        out["gnn_edge_types"] = gnn_edge_types

    return out


def _collate_binaries(
    binary_list: list[dict],
    *,
    include_function_features: bool = True,
) -> dict[str, torch.Tensor]:
    """Pad tokens and optionally stack 132-dimensional function features."""
    if not binary_list:
        raise ValueError("cannot collate an empty binary list")

    sequences = []
    type_sequences = []
    mask_sequences = []
    for index, binary in enumerate(binary_list):
        token_ids = torch.as_tensor(binary.get("token_ids", []), dtype=torch.long)
        if token_ids.ndim != 1 or token_ids.numel() == 0:
            raise ValueError(f"binary {index} must contain a non-empty 1D token sequence")
        token_type_ids = binary.get("token_type_ids")
        attention_mask = binary.get("attention_mask")
        token_type_ids = (
            torch.zeros_like(token_ids)
            if token_type_ids is None
            else torch.as_tensor(token_type_ids, dtype=torch.long)
        )
        attention_mask = (
            torch.ones_like(token_ids)
            if attention_mask is None
            else torch.as_tensor(attention_mask, dtype=torch.long)
        )
        if token_type_ids.shape != token_ids.shape or attention_mask.shape != token_ids.shape:
            raise ValueError(
                f"binary {index} token, token-type, and attention-mask shapes must match"
            )
        sequences.append(token_ids)
        type_sequences.append(token_type_ids)
        mask_sequences.append(attention_mask)

    max_len = max((len(s) for s in sequences), default=1)

    input_ids = torch.full((len(binary_list), max_len), PAD_TOKEN_ID, dtype=torch.long)
    token_type_ids = torch.full_like(input_ids, PAD_TOKEN_ID)
    attention_mask = torch.zeros_like(input_ids)

    for row, (ids, types, mask) in enumerate(
        zip(sequences, type_sequences, mask_sequences, strict=False)
    ):
        seq_len = len(ids)
        input_ids[row, :seq_len] = ids
        token_type_ids[row, :seq_len] = types
        attention_mask[row, :seq_len] = mask

    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }
    if include_function_features:
        function_features = [
            torch.as_tensor(
                binary.get("function_features", [0.0] * 132),
                dtype=torch.float32,
            )
            for binary in binary_list
        ]
        for index, features in enumerate(function_features):
            if features.shape != (132,):
                raise ValueError(
                    "binary "
                    f"{index} must contain 132 function features, got {tuple(features.shape)}"
                )
        out["function_features"] = torch.stack(function_features)
    return out
