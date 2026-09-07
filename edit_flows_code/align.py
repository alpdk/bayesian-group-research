"""Sequence alignment utilities for Edit Flows (X-space <-> Z-space).

Ported from https://github.com/TheMatrixMaster/edit-flows-demo (utils.py) with
special tokens passed explicitly instead of module-level constants.
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F


def _align_pair(seq_0: torch.Tensor, seq_1: torch.Tensor, gap_token: int) -> Tuple[List[int], List[int]]:
    """Minimum-edit-distance alignment of two sequences via dynamic programming."""
    seq_0, seq_1 = seq_0.cpu().numpy(), seq_1.cpu().numpy()
    m, n = len(seq_0), len(seq_1)

    dp = [[i + j if i == 0 or j == 0 else 0 for j in range(n + 1)] for i in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if seq_0[i-1] == seq_1[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    aligned_0, aligned_1 = [], []
    i, j = m, n
    while i or j:
        if i and j and seq_0[i-1] == seq_1[j-1]:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(seq_1[j-1])
            i, j = i-1, j-1
        elif i and j and dp[i][j] == dp[i-1][j-1] + 1:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(seq_1[j-1])
            i, j = i-1, j-1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            aligned_0.append(seq_0[i-1])
            aligned_1.append(gap_token)
            i -= 1
        else:
            aligned_0.append(gap_token)
            aligned_1.append(seq_1[j-1])
            j -= 1

    return aligned_0[::-1], aligned_1[::-1]


def opt_align_xs_to_zs(
    x_0: torch.Tensor, x_1: torch.Tensor, gap_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aligns x_0 and x_1 to equal-length Z-space sequences using min edit distance."""
    aligned_pairs = [_align_pair(x_0[b], x_1[b], gap_token) for b in range(x_0.shape[0])]
    x_0_aligned = torch.stack(
        [torch.tensor(pair[0], dtype=x_0.dtype, device=x_0.device) for pair in aligned_pairs])
    x_1_aligned = torch.stack(
        [torch.tensor(pair[1], dtype=x_1.dtype, device=x_1.device) for pair in aligned_pairs])
    return x_0_aligned, x_1_aligned


def rm_gap_tokens(
    z: torch.Tensor, gap_token: int, pad_token: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Removes gap tokens from a batched Z tensor and right-pads with pad_token."""
    batch_size, _ = z.shape
    z_no_gap = []
    for b in range(batch_size):
        z_no_pad = z[b][z[b] != pad_token]
        z_no_gap.append(z_no_pad[z_no_pad != gap_token])
    max_len = max(len(zi) for zi in z_no_gap)
    x = torch.stack([F.pad(zi, (0, max_len - len(zi)), value=pad_token) for zi in z_no_gap], dim=0).long()
    x_pad_mask = (x == pad_token)
    z_gap_mask = (z == gap_token)
    z_pad_mask = (z == pad_token)
    assert ((~x_pad_mask).sum(1) + z_gap_mask.sum(1)).equal((~z_pad_mask).sum(1))
    return x, x_pad_mask, z_gap_mask, z_pad_mask


def make_ut_mask_from_z(
    z_t: torch.Tensor,
    z_1: torch.Tensor,
    vocab_size: int,
    gap_token: int,
    pad_token: int,
) -> torch.Tensor:
    """
    Mask over the (2 * vocab_size + 1) edit operations indicating, for every
    Z-position where z_t and z_1 differ, the single operation that brings z_t
    closer to z_1:

    - z_t[i] = GAP, z_1[i] = c   -> insert token c   (slot c)
    - z_t[i] = c,   z_1[i] = GAP -> delete           (last slot)
    - z_t[i] = c1,  z_1[i] = c2  -> substitute to c2 (slot vocab_size + c2)
    """
    batch_size, z_seq_len = z_t.shape
    n_ops = 2 * vocab_size + 1

    z_neq = (z_t != z_1) & (z_t != pad_token) & (z_1 != pad_token)
    z_ins = (z_t == gap_token) & (z_1 != gap_token) & z_neq
    z_del = (z_t != gap_token) & (z_1 == gap_token) & z_neq
    z_sub = z_neq & ~z_ins & ~z_del

    u_mask = torch.zeros((batch_size, z_seq_len, n_ops), dtype=torch.bool, device=z_t.device)
    u_mask[z_ins, z_1[z_ins]] = True
    u_mask[z_sub, z_1[z_sub] + vocab_size] = True
    u_mask[:, :, -1][z_del] = True

    assert z_neq.sum() == (z_ins | z_del | z_sub).sum(), "Mismatch in number of edits"
    assert z_neq.sum() == u_mask.sum(), "Mismatch in number of edits in mask"

    return u_mask


def fill_gap_tokens_with_repeats(
    x_ut: torch.Tensor,
    z_gap_mask: torch.Tensor,
    z_pad_mask: torch.Tensor,
) -> torch.Tensor:
    """Expands per-X-position rates back to Z-space by repeating the last
    non-gap position's rates at gap positions."""
    batch_size, _ = z_gap_mask.shape
    _, x_seq_len, _ = x_ut.shape

    non_gap_mask = ~z_gap_mask
    indices = non_gap_mask.cumsum(dim=1) - 1
    indices = indices.clamp(min=0, max=x_seq_len - 1)

    batch_indices = torch.arange(batch_size, device=x_ut.device).unsqueeze(1)
    result = x_ut[batch_indices, indices]
    result[z_pad_mask] = 0
    return result
