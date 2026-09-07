"""Prompt-conditional Edit Flows transformer for code generation.

Adapted from the SimpleEditFlowsTransformer of
https://github.com/TheMatrixMaster/edit-flows-demo: the question (prompt) is
encoded as a prefix segment that the model attends to but never edits; the
edit-operation heads (insert/substitute/delete rates and token distributions)
are only produced for the generated part x_t.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (batch_size, 1)

        half_dim = self.hidden_dim // 2
        t32 = t.float()
        emb = torch.log(torch.tensor(10000.0, device=t.device, dtype=torch.float32)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -emb)
        emb = t32 * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)

        return emb.to(dtype=t.dtype)  # (batch_size, hidden_dim)


class CondEditFlowsTransformer(nn.Module):
    """Bidirectional transformer over [prompt ; x_t] that outputs edit rates for x_t."""

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 8,
        max_seq_len: int = 1024,
        bos_token_id: int = 256,
        pad_token_id: int = 257,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        assert bos_token_id < vocab_size, "bos_token_id must be less than vocab_size"
        assert pad_token_id < vocab_size, "pad_token_id must be less than vocab_size"

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_dim)
        self.segment_embedding = nn.Embedding(2, hidden_dim)  # 0 = prompt, 1 = x_t
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu',
                batch_first=True)
            for _ in range(num_layers)
        ])
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        self.rates_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),  # insert, substitute, delete rates
        )
        self.ins_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self.sub_logits_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, vocab_size),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        prompt: torch.Tensor,          # (batch, p_len) long
        prompt_pad_mask: torch.Tensor, # (batch, p_len) bool
        tokens: torch.Tensor,          # (batch, x_len) long
        time_step: torch.Tensor,       # (batch, 1) float
        padding_mask: torch.Tensor,    # (batch, x_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (rates, ins_probs, sub_probs) for the x_t segment only."""
        batch_size, p_len = prompt.shape
        _, x_len = tokens.shape
        assert p_len <= self.max_seq_len and x_len <= self.max_seq_len, \
            f"sequence too long for positional table ({p_len=}, {x_len=}, max={self.max_seq_len})"
        device = tokens.device

        p_positions = torch.arange(p_len, device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_emb = (
            self.token_embedding(prompt)
            + self.pos_embedding(p_positions)
            + self.segment_embedding.weight[0]
        )

        time_emb = self.time_embedding(time_step)  # (batch, hidden_dim)
        x_positions = torch.arange(x_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x_emb = (
            self.token_embedding(tokens)
            + self.pos_embedding(x_positions)
            + self.segment_embedding.weight[1]
            + time_emb.unsqueeze(1)
        )

        h = torch.cat([prompt_emb, x_emb], dim=1)                       # (batch, p_len + x_len, hidden)
        key_pad_mask = torch.cat([prompt_pad_mask, padding_mask], dim=1)
        for layer in self.layers:
            h = layer(h, src_key_padding_mask=key_pad_mask)

        h = self.final_layer_norm(h[:, p_len:])                         # (batch, x_len, hidden)
        # bf16 softmax/softplus can overflow; run output heads in fp32.
        h32 = h.float()

        def _head_fp32(head: nn.Module, x: torch.Tensor) -> torch.Tensor:
            y = x
            for layer in head:
                if isinstance(layer, nn.Linear):
                    bias = None if layer.bias is None else layer.bias.float()
                    y = F.linear(y, layer.weight.float(), bias)
                else:
                    y = layer(y)
            return y

        ins_logits = _head_fp32(self.ins_logits_out, h32).clamp(-30, 30)
        sub_logits = _head_fp32(self.sub_logits_out, h32).clamp(-30, 30)
        rate_logits = _head_fp32(self.rates_out, h32).clamp(-20, 20)
        rates = F.softplus(rate_logits)
        ins_probs = F.softmax(ins_logits, dim=-1)
        sub_probs = F.softmax(sub_logits, dim=-1)

        mask_expanded = (~padding_mask).unsqueeze(-1).to(rates.dtype)
        rates = rates * mask_expanded
        ins_probs = ins_probs * mask_expanded
        sub_probs = sub_probs * mask_expanded
        rates = torch.nan_to_num(rates, nan=0.0, posinf=1e4, neginf=0.0)
        ins_probs = torch.nan_to_num(ins_probs, nan=0.0, posinf=1.0, neginf=0.0)
        sub_probs = torch.nan_to_num(sub_probs, nan=0.0, posinf=1.0, neginf=0.0)

        out_dtype = h.dtype
        return rates.to(out_dtype), ins_probs.to(out_dtype), sub_probs.to(out_dtype)


def save_checkpoint(path, model: CondEditFlowsTransformer, optim: torch.optim.Optimizer, extra: dict | None = None):
    payload = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'vocab_size': model.vocab_size,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'num_heads': model.num_heads,
        'max_seq_len': model.max_seq_len,
        'bos_token_id': model.bos_token_id,
        'pad_token_id': model.pad_token_id,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, device) -> tuple[CondEditFlowsTransformer, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = CondEditFlowsTransformer(
        vocab_size=checkpoint['vocab_size'],
        hidden_dim=checkpoint['hidden_dim'],
        num_layers=checkpoint['num_layers'],
        num_heads=checkpoint['num_heads'],
        max_seq_len=checkpoint['max_seq_len'],
        bos_token_id=checkpoint['bos_token_id'],
        pad_token_id=checkpoint['pad_token_id'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.to(device), checkpoint
