"""Canonical model construction for training and evaluation."""

import torch
from omegaconf import DictConfig

from model.latent_mdm import CombinedTransformer
from model.transformer import MDMConfig, MDMTransformer


COMBINED_STRATEGIES = frozenset({"latmdm"})


def build_model(
    cfg: DictConfig,
    device: torch.device,
    *,
    is_main: bool = False,
) -> tuple[torch.nn.Module, MDMConfig]:
    """Build the configured architecture and apply optional ARM initialization."""
    strategy = cfg.training.strategy
    model_cfg = cfg.model
    if strategy in COMBINED_STRATEGIES:
        encoder_config = MDMConfig(**model_cfg.encoder)
        planner_config = MDMConfig(**model_cfg.planner)
        decoder_config = MDMConfig(**model_cfg.decoder)
        effective_config = planner_config
        model = CombinedTransformer(
            encoder_config,
            planner_config,
            decoder_config,
            tie_embeddings=bool(model_cfg.get("tie_embeddings", False)),
        ).to(device)
    else:
        effective_config = MDMConfig(**model_cfg)
        model = MDMTransformer(effective_config).to(device)

    arm_init_path = (
        model_cfg.get("arm_init", "none")
        if strategy not in COMBINED_STRATEGIES
        else "none"
    )
    if arm_init_path != "none":
        effective_config.predict_next_token = True
        if is_main:
            print(f"Initializing MDM from ARM checkpoint: {arm_init_path}")
        arm_ckpt = torch.load(arm_init_path, map_location="cpu")
        state_dict = arm_ckpt.get("model_state_dict", arm_ckpt)
        model.load_state_dict(state_dict, strict=True)

    return model, effective_config
