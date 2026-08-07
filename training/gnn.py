"""Training loop for structural-prefix models."""

from __future__ import annotations

import torch

from dataset.collate import apply_modality_dropout
from training.runtime import accumulation_window_size, to_device
from training.struct_heads import StructHeadsOnTransformer, struct_loss


def train_epoch(
    model: torch.nn.Module,
    loader,
    contrastive_loss_fn: torch.nn.Module,
    struct_heads: StructHeadsOnTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    trainable_params: list[torch.nn.Parameter],
    *,
    grad_clip: float,
    lambda_contrastive: float,
    lambda_struct: float,
    scaler: torch.amp.GradScaler | None,
    grad_accum_steps: int,
    modality_dropout_ratio: float,
    modality_dropout_side: str,
    modality_dropout_drop_features: bool,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float, float]:
    """Train one epoch and return total, contrastive, and structural loss."""
    model.train()
    struct_heads.train()
    totals = [0.0, 0.0, 0.0]
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(loader)

    for step, batch in enumerate(loader):
        batch = to_device(batch, device)
        if modality_dropout_ratio > 0:
            apply_modality_dropout(
                batch,
                ratio=modality_dropout_ratio,
                side=modality_dropout_side,
                drop_function_features=modality_dropout_drop_features,
            )

        with torch.amp.autocast(device.type, enabled=use_amp, dtype=amp_dtype):
            embeddings_a, embeddings_b, pooled_a, pooled_b = model.get_embedding_pairs_with_pooled(
                batch
            )
            contrastive_loss = contrastive_loss_fn(embeddings_a, embeddings_b)

        predictions_a = struct_heads(pooled_a.float())
        predictions_b = struct_heads(pooled_b.float())
        batch_size = embeddings_a.shape[0]
        struct_a, _ = struct_loss(
            predictions_a,
            batch["edge_index_a"],
            batch["edge_types_a"],
            batch["graph_batch_a"],
            batch_size,
        )
        struct_b, _ = struct_loss(
            predictions_b,
            batch["edge_index_b"],
            batch["edge_types_b"],
            batch["graph_batch_b"],
            batch_size,
        )
        structural_loss = (struct_a + struct_b) * 0.5
        full_loss = lambda_contrastive * contrastive_loss + lambda_struct * structural_loss
        window_size = accumulation_window_size(step, num_batches, grad_accum_steps)
        loss = full_loss / window_size
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()

        at_boundary = (step + 1) % grad_accum_steps == 0 or step + 1 == num_batches
        if at_boundary:
            optimizer_updated = True
            if scaler is None:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                optimizer.step()
            else:
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_updated = scaler.get_scale() >= previous_scale
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None and optimizer_updated:
                scheduler.step()

        totals[0] += full_loss.item()
        totals[1] += contrastive_loss.item()
        totals[2] += structural_loss.item()

    if num_batches == 0:
        raise ValueError("Cannot train on an empty data loader")
    return tuple(total / num_batches for total in totals)
