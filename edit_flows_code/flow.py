"""Couplings and kappa schedulers for Edit Flows.

Ported from https://github.com/TheMatrixMaster/edit-flows-demo (flow.py) with
the sine-wave specifics removed and special tokens made explicit.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor


def x2prob(x: Tensor, vocab_size: int) -> Tensor:
    """Converts a token sequence to its one-hot class distribution."""
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def sample_p(pt: Tensor, temperature: float = 1.0) -> Tensor:
    """Samples token sequences from a class distribution (batch, len, classes)."""
    b, l, c = pt.shape
    xt = torch.multinomial(pt.reshape(b * l, c) / temperature, 1)
    return xt.reshape(b, l)


class Coupling(ABC):
    @abstractmethod
    def sample(self, x1: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError


class EmptyCoupling(Coupling):
    """Samples empty prior sequences (pure insertion-based generation)."""

    def sample(self, x1: Tensor):
        x0 = torch.empty((x1.shape[0], 0), dtype=x1.dtype, device=x1.device).long()
        return x0, x1


class ExtendedCoupling(Coupling):
    """Randomly inserts extra tokens into the target sequence."""

    def __init__(self, n_insert: int, vocab_size: int, pad_token: int):
        self.n_insert = n_insert
        self.vocab_size = vocab_size
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, x1_seq_len = x1.shape
        x1_pad_mask = (x1 == self.pad_token)
        x1_seq_lengths = (~x1_pad_mask).sum(dim=1).tolist()

        ins_positions = torch.stack([
            torch.randint(0, seqlen + 1, size=(self.n_insert,), dtype=torch.long, device=x1.device)
            for seqlen in x1_seq_lengths
        ])
        ins_positions = torch.sort(ins_positions, dim=1)[0]     # (batch_size, n_insert)

        max_new_len = self.n_insert + x1_seq_len
        x0 = torch.full((batch_size, max_new_len), self.pad_token, dtype=x1.dtype, device=x1.device)

        batch_indices = torch.arange(batch_size, device=x1.device).unsqueeze(1)
        orig_positions = torch.arange(x1_seq_len, device=x1.device).unsqueeze(0).expand(batch_size, -1)

        num_insert_before = (ins_positions.unsqueeze(2) <= orig_positions.unsqueeze(1)).sum(dim=1)
        new_orig_positions = orig_positions + num_insert_before
        x0[batch_indices, new_orig_positions] = x1

        ins_new_positions = ins_positions + torch.arange(self.n_insert, device=x1.device).unsqueeze(0)
        ins_tokens = torch.randint(0, self.vocab_size, size=(batch_size, self.n_insert), dtype=x1.dtype, device=x1.device)
        x0[batch_indices, ins_new_positions] = ins_tokens

        return x0, x1


class UniformCoupling(Coupling):
    """Samples uniform-random prior sequences within a length range."""

    def __init__(self, min_len: int, max_len: int, vocab_size: int, pad_token: int, mirror_len: bool = False):
        self.min_len = min_len
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.mirror_len = mirror_len
        self.pad_token = pad_token

    def sample(self, x1: Tensor):
        batch_size, _ = x1.shape
        x1_pad_mask = (x1 == self.pad_token)
        if self.mirror_len:
            x0_pad_mask = x1_pad_mask
            x0_max_len = x1.shape[1]
        else:
            x0_seq_len = torch.randint(self.min_len, self.max_len + 1, size=(batch_size,)).long()
            x0_max_len = int(x0_seq_len.max().item())
            x0_pad_mask = torch.arange(x0_max_len, device=x1.device).expand(batch_size, -1) >= x0_seq_len.unsqueeze(1)

        x0 = torch.randint(0, self.vocab_size, size=(batch_size, x0_max_len), dtype=x1.dtype, device=x1.device)
        x0[x0_pad_mask] = self.pad_token
        return x0, x1


class KappaScheduler(ABC):
    @abstractmethod
    def __call__(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, t: Tensor) -> Tensor:
        raise NotImplementedError


class CubicScheduler(KappaScheduler):
    def __init__(self, a: float = 2.0, b: float = 0.5) -> None:
        self.a = a
        self.b = b

    def __call__(self, t: Tensor) -> Tensor:
        return -2 * (t**3) + 3 * (t**2) + self.a * (t**3 - 2 * t**2 + t) + self.b * (t**3 - t**2)

    def derivative(self, t: Tensor) -> Tensor:
        return -6 * (t**2) + 6 * t + self.a * (3 * t**2 - 4 * t + 1) + self.b * (3 * t**2 - 2 * t)
