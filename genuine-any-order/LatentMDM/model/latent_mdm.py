"""LatentMDM transformer variants and combined model behavior."""

from dataclasses import replace

import torch
import torch.nn as nn

from model.transformer import MDMConfig, MDMTransformer


def first_eos_valid_mask(tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
    """Include tokens through the first EOS and exclude any padded suffix."""
    is_eos = tokens == eos_id
    any_eos = is_eos.any(dim=1)
    first_eos = is_eos.float().argmax(dim=1)
    first_eos = torch.where(
        any_eos,
        first_eos,
        torch.full_like(first_eos, tokens.shape[1] - 1),
    )
    positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
    return positions <= first_eos.unsqueeze(1)


class LatentMDMTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert (
            config.input_type == "continuous"
        ), "LatentMDMTransformer only supports continuous inputs"
        self.prompt_emb = nn.Embedding(config.vocab_size, config.hidden_size)


class EncodeTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert (
            config.input_type == "discrete"
        ), "EncodeTransformer only supports discrete inputs"


class DecodeTransformer(MDMTransformer):
    def __init__(self, config: MDMConfig):
        super().__init__(config)
        assert (
            config.input_type == "discrete"
        ), "DecodeTransformer only supports discrete inputs"
        assert config.causal, "DecodeTransformer must be initialized with causal=True"

    def forward(
        self,
        input_ids: torch.Tensor,
        planner_condition: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _ = input_ids.shape
        if planner_condition.dim() == 2:
            planner_condition = planner_condition.unsqueeze(1)
        if planner_condition.shape[:2] != (batch_size, 1):
            raise ValueError(
                "planner_condition must have shape (B, D) or (B, 1, D), "
                f"got {tuple(planner_condition.shape)} for batch {batch_size}"
            )

        x = self.emb(input_ids)
        x = torch.cat([planner_condition, x], dim=1)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.lm_head(x)


class CombinedTransformer(nn.Module):
    def __init__(
        self,
        encoder_config: MDMConfig,
        planner_config: MDMConfig,
        decoder_config: MDMConfig,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        assert (
            encoder_config.return_last_hidden_states
        ), "LatentMDM encoder must return hidden states"
        assert (
            planner_config.return_last_hidden_states
        ), "LatentMDM planner must return hidden states"
        assert (
            not decoder_config.return_last_hidden_states
        ), "LatentMDM decoder must return logits"

        use_cls = str(getattr(encoder_config, "segment_pooling", "mean")) == "cls"
        if use_cls:
            self.cls_token_id = int(encoder_config.vocab_size)
            encoder_config = replace(
                encoder_config,
                vocab_size=encoder_config.vocab_size + 1,
            )
            if tie_embeddings:
                planner_config = replace(
                    planner_config,
                    vocab_size=planner_config.vocab_size + 1,
                )
                decoder_config = replace(
                    decoder_config,
                    vocab_size=decoder_config.vocab_size + 1,
                )
        else:
            self.cls_token_id = None

        self.encoder = EncodeTransformer(encoder_config)
        self.planner = LatentMDMTransformer(planner_config)
        self.decoder = DecodeTransformer(decoder_config)
        if tie_embeddings:
            if not (
                encoder_config.vocab_size
                == planner_config.vocab_size
                == decoder_config.vocab_size
                and encoder_config.hidden_size
                == planner_config.hidden_size
                == decoder_config.hidden_size
            ):
                raise ValueError(
                    "tie_embeddings requires encoder, planner, and decoder to share "
                    "vocab_size and hidden_size."
                )
            self.planner.prompt_emb.weight = self.encoder.emb.weight
            self.decoder.emb.weight = self.encoder.emb.weight
            if isinstance(self.decoder.lm_head, nn.Linear):
                self.decoder.lm_head.weight = self.encoder.emb.weight

        self.mask_token = nn.Parameter(
            torch.randn(planner_config.hidden_size) * 0.02
        )
        if encoder_config.hidden_size != planner_config.hidden_size:
            self.encoder_to_planner = nn.Linear(
                encoder_config.hidden_size,
                planner_config.hidden_size,
            )
        else:
            self.encoder_to_planner = nn.Identity()

    def encode_segments(
        self,
        segments: torch.Tensor,
        eos_id: int,
    ) -> torch.Tensor:
        """Encode answer segments into planner-space latent representations."""
        segment_pooling = str(
            getattr(self.encoder.config, "segment_pooling", "mean")
        )
        if segment_pooling == "mean":
            hidden = self.encoder(segments)
            pooled = hidden.mean(dim=1)
        elif segment_pooling == "eos_mean":
            valid = first_eos_valid_mask(segments, int(eos_id))
            hidden = self.encoder(segments, attn_mask=valid[:, None, None, :])
            valid_f = valid.to(hidden.dtype)
            pooled = (hidden * valid_f.unsqueeze(-1)).sum(dim=1) / valid_f.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(1.0)
        elif segment_pooling == "cls":
            cls_id = int(self.cls_token_id)
            cls_col = segments.new_full((segments.shape[0], 1), cls_id)
            cls_segments = torch.cat([cls_col, segments], dim=1)
            valid = first_eos_valid_mask(segments, int(eos_id))
            cls_valid = torch.ones(
                valid.shape[0],
                1,
                dtype=torch.bool,
                device=valid.device,
            )
            valid_full = torch.cat([cls_valid, valid], dim=1)
            hidden = self.encoder(
                cls_segments,
                attn_mask=valid_full[:, None, None, :],
            )
            pooled = hidden[:, 0]
        else:
            raise ValueError(
                "model.encoder.segment_pooling must be one of: mean, eos_mean, cls; "
                f"got {segment_pooling!r}."
            )
        return self.encoder_to_planner(pooled)

    def forward(
        self,
        prompt_ids: torch.Tensor,
        split_labels: torch.Tensor,
        prompt_len: torch.Tensor,
        eos_id: int,
        decoder_chunk_size: int = 1024,
        slot_reweight: bool = False,
    ) -> torch.Tensor:
        """Route DDP and direct model calls to the canonical training objective."""
        from training.losses import latmdm_loss

        return latmdm_loss(
            self,
            prompt_ids,
            split_labels,
            prompt_len,
            eos_id=eos_id,
            decoder_chunk_size=decoder_chunk_size,
            slot_reweight=slot_reweight,
        )
