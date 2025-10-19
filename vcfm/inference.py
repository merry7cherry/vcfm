from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import (
    CallbackConfig,
    Config,
    DatasetConfig,
    ModelConfig,
    NetworkConfig,
    TrainingConfig,
)
from .model_factory import build_model


def load_checkpoint(
    path: str | Path,
    *,
    device: Optional[torch.device] = None,
) -> Tuple[Config, torch.nn.Module, Optional[torch.nn.Module]]:
    checkpoint = torch.load(path, map_location="cpu")
    cfg_dict = checkpoint.get("config")
    if cfg_dict is None:
        raise ValueError("Checkpoint does not include configuration data.")
    cfg = Config(
        dataset=DatasetConfig(**cfg_dict["dataset"]),
        model=ModelConfig(**cfg_dict["model"]),
        network=NetworkConfig(**cfg_dict["network"]),
        training=TrainingConfig(**cfg_dict.get("training", {})),
        callbacks=CallbackConfig(**cfg_dict.get("callbacks", {})),
    )
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model"])
    target_device = device or torch.device("cpu")
    model.to(target_device)
    ema_state = checkpoint.get("ema")
    ema_model = None
    if ema_state:
        ema_model = build_model(cfg)
        ema_model.load_state_dict(ema_state)
        ema_model.to(target_device)
    return cfg, model, ema_model


def generate(
    model: torch.nn.Module,
    *,
    sample_shape,
    n_iters: int,
    device: Optional[torch.device] = None,
    class_labels: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    return model.sample(sample_shape, n_iters, device, class_labels=class_labels)


__all__ = ["load_checkpoint", "generate"]
