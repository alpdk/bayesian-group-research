"""Training objectives shared by training and validation."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from model.latent_mdm import first_eos_valid_mask


def grad_norm(parameters) -> float:
    """Return the aggregate L2 gradient norm."""
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            param_norm = parameter.grad.detach().norm(p=2).item()
            total += param_norm**2
    return total**0.5


def decoder_ce_loss(
    model,
    planner_conditions: torch.Tensor,
    target_ids: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int,
    slot_weights: Optional[torch.Tensor] = None,
    batch_size: Optional[int] = None,
) -> torch.Tensor:
    """Compute chunked autoregressive decoder loss for masked latent slots."""
    valid = first_eos_valid_mask(target_ids, eos_id)
    total_loss = planner_conditions.new_zeros(())
    if slot_weights is None:
        total_count = planner_conditions.new_zeros(())
    else:
        slot_weights = slot_weights.to(
            device=planner_conditions.device,
            dtype=total_loss.dtype,
        )
        if slot_weights.shape[0] != target_ids.shape[0]:
            raise ValueError(
                "slot_weights must have one weight per decoder target segment."
            )
        if batch_size is None:
            raise ValueError("batch_size is required when slot_weights is set.")
        batch_count = total_loss.new_tensor(float(batch_size)).clamp(min=1.0)

    for start in range(0, target_ids.shape[0], decoder_chunk_size):
        end = min(start + decoder_chunk_size, target_ids.shape[0])
        chunk_targets = target_ids[start:end]
        chunk_valid = valid[start:end]
        if not chunk_valid.any():
            continue
        logits = model.decoder(chunk_targets, planner_conditions[start:end])
        pred_logits = logits[:, :-1, :]
        token_loss = F.cross_entropy(
            pred_logits[chunk_valid],
            chunk_targets[chunk_valid],
            reduction="none" if slot_weights is not None else "sum",
        )
        if slot_weights is None:
            total_loss = total_loss + token_loss
            total_count = total_count + chunk_valid.sum().to(total_loss.dtype)
        else:
            valid_slot_idx = chunk_valid.nonzero(as_tuple=True)[0]
            chunk_slot_loss = total_loss.new_zeros((end - start,))
            chunk_slot_loss.index_add_(
                0,
                valid_slot_idx,
                token_loss.to(total_loss.dtype),
            )
            total_loss = total_loss + (
                chunk_slot_loss * slot_weights[start:end]
            ).sum()

    if slot_weights is None:
        return total_loss / total_count.clamp(min=1.0)
    return total_loss / batch_count


def mdm_loss(
    model,
    input_ids: torch.Tensor,
    mask_id: int,
    prompt_mask: Optional[torch.Tensor] = None,
    arm_init: bool = False,
) -> torch.Tensor:
    """Compute the masked diffusion language-model objective."""
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    device = input_ids.device
    batch_size, length = input_ids.shape
    effective_length = length - prompt_mask.sum(dim=1, keepdim=True)
    num_mask = (
        torch.floor(
            torch.rand(batch_size, 1, device=device) * effective_length.clamp(min=1)
        ).long()
        + 1
    )

    scores = torch.rand((batch_size, length), device=device).masked_fill(
        prompt_mask, float("inf")
    ).argsort(dim=1)
    order = scores.argsort(dim=1)
    mask_indices = order < num_mask
    masked_input = torch.where(mask_indices, mask_id, input_ids)
    logits = model(masked_input)

    num_mask = num_mask.float().expand_as(mask_indices)
    if arm_init:
        ce = F.cross_entropy(
            logits[:, :-1, :][mask_indices[:, 1:]],
            input_ids[:, 1:][mask_indices[:, 1:]],
            reduction="none",
        )
    else:
        ce = F.cross_entropy(
            logits[mask_indices], input_ids[mask_indices], reduction="none"
        )
    loss = ce / num_mask[mask_indices]
    return loss.sum() / batch_size


def arm_loss(
    model,
    input_ids: torch.Tensor,
    eos_id: int,
    prompt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute next-token loss through the first EOS after the prompt."""
    if prompt_mask is None:
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    logits = model(input_ids)
    targets = input_ids[:, 1:]
    pred_logits = logits[:, :-1, :]
    valid = ~prompt_mask[:, 1:]

    if eos_id is not None:
        is_eos = targets == eos_id
    else:
        is_eos = torch.zeros_like(targets, dtype=torch.bool)
    any_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1)
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, targets.shape[1] - 1),
    )

    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    valid = valid & (positions <= first_eos.unsqueeze(1))
    if valid.sum().item() == 0:
        return pred_logits.sum() * 0.0
    return F.cross_entropy(pred_logits[valid], targets[valid], reduction="mean")


def latmdm_loss(
    model,
    prompt: torch.Tensor,
    split_labels: torch.Tensor,
    prompt_len: torch.Tensor,
    eos_id: int,
    decoder_chunk_size: int = 1024,
    slot_reweight: bool = False,
) -> torch.Tensor:
    """Compute the LatentMDM masked-slot training objective."""
    if isinstance(model, DDP):
        return model(
            prompt,
            split_labels,
            prompt_len,
            eos_id=int(eos_id),
            decoder_chunk_size=int(decoder_chunk_size),
            slot_reweight=bool(slot_reweight),
        )

    batch_size, max_prompt_len = prompt.shape
    _, max_seg_num, _ = split_labels.shape
    planner_len = max_prompt_len + max_seg_num
    if planner_len > model.planner.config.max_position:
        raise ValueError(
            f"planner sequence length {planner_len} exceeds max_position "
            f"{model.planner.config.max_position}"
        )

    device = prompt.device
    prompt_len = prompt_len.to(
        device=device,
        dtype=torch.long,
    ).clamp(0, max_prompt_len)

    prompt_aug = prompt.new_full((batch_size, planner_len), int(eos_id))
    prompt_aug[:, :max_prompt_len] = prompt
    planner_inputs = model.planner.prompt_emb(prompt_aug).clone()

    segment_is_real = ~(split_labels == int(eos_id)).all(dim=2)
    if segment_is_real.any():
        flat_segments = split_labels[segment_is_real]
        flat_latents = model.encode_segments(
            flat_segments,
            int(eos_id),
        ).to(planner_inputs.dtype)
        batch_idx, seg_idx = segment_is_real.nonzero(as_tuple=True)
        slot_pos = prompt_len[batch_idx] + seg_idx
        planner_inputs[batch_idx, slot_pos] = flat_latents

    num_mask = (
        torch.floor(
            torch.rand(batch_size, 1, device=device) * max_seg_num
        ).long()
        + 1
    )
    segment_scores = torch.rand((batch_size, max_seg_num), device=device)
    segment_order = segment_scores.argsort(dim=1).argsort(dim=1)
    segment_mask = segment_order < num_mask

    slot_pos = prompt_len.unsqueeze(1) + torch.arange(
        max_seg_num,
        device=device,
    ).unsqueeze(0)
    batch_slots = torch.arange(batch_size, device=device).unsqueeze(1).expand(
        batch_size,
        max_seg_num,
    )
    mask_indices = torch.zeros(
        (batch_size, planner_len),
        dtype=torch.bool,
        device=device,
    )
    mask_indices[batch_slots, slot_pos] = segment_mask

    mask_token = model.mask_token.to(dtype=planner_inputs.dtype).view(1, 1, -1)
    planner_inputs = torch.where(
        mask_indices.unsqueeze(-1),
        mask_token,
        planner_inputs,
    )
    planner_out = model.planner(planner_inputs)

    masked_batch, masked_pos = mask_indices.nonzero(as_tuple=True)
    if masked_batch.numel() == 0:
        return planner_out.sum() * 0.0

    planner_conditions = planner_out[masked_batch, masked_pos]
    tail_idx = masked_pos - prompt_len[masked_batch]
    target_ids = split_labels[masked_batch, tail_idx]
    slot_weights = None
    if slot_reweight:
        num_mask_per_sample = num_mask.squeeze(1).to(
            dtype=planner_conditions.dtype
        )
        slot_weights = 1.0 / num_mask_per_sample[masked_batch]

    return decoder_ce_loss(
        model,
        planner_conditions,
        target_ids,
        int(eos_id),
        max(1, int(decoder_chunk_size)),
        slot_weights=slot_weights,
        batch_size=batch_size if slot_reweight else None,
    )


def get_prompt_len(batch, device: torch.device) -> torch.Tensor:
    """Read prompt lengths from either supported batch representation."""
    if "prompt_mask" in batch:
        return batch["prompt_mask"].to(device).sum(dim=1).long()
    return batch["prompt_len"].to(device)
