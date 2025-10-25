from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import OmegaConf


@dataclass
class DatasetConfig:
    name: str
    img_resolution: int
    in_channels: int
    out_channels: int
    label_dim: int
    batch_size: int
    num_workers: int
    data_dir: str
    sample_shape: List[int]
    plot_type: str
    fid_sample_shape: List[int]
    n_dataset_samples: int


@dataclass
class ModelConfig:
    name: str
    use_ema: bool
    ema_rate: float
    ema_type: str
    class_conditional: bool
    total_training_steps: int
    velocity_learning_rate: float
    phi_learning_rate: float
    velocity_weight_decay: float
    phi_weight_decay: float
    sigma_min: float
    sigma_max: float
    flow_matching_theta_weight: float
    straightness_weight: float
    kl_phi_weight: float
    latent_dim: int
    phi_hidden_channels: int
    phi_num_layers: int


@dataclass
class NetworkConfig:
    name: str
    embedding_type: Optional[str] = None
    encoder_type: Optional[str] = None
    decoder_type: Optional[str] = None
    channel_mult_noise: Optional[float] = None
    resample_filter: Optional[List[int]] = None
    model_channels: int = 128
    channel_mult: Optional[List[int]] = None
    dropout: float = 0.0
    dropout_res: Optional[int] = None
    num_blocks: int = 3
    reload_url: str = ""


@dataclass
class TrainingConfig:
    log_every: int = 100
    eval_every: int = 1000
    sample_every: int = 1000
    sample_steps: int = 50
    checkpoint_every: int = 0
    grad_clip: Optional[float] = None
    output_dir: str = "runs"
    seed: int = 42


@dataclass
class CallbackConfig:
    generate: bool = True
    fid: bool = True


@dataclass
class Config:
    dataset: DatasetConfig
    model: ModelConfig
    network: NetworkConfig
    training: TrainingConfig
    callbacks: CallbackConfig


def _compose_from_defaults(config_path: Path) -> Dict[str, Any]:
    cfg = OmegaConf.load(config_path)
    defaults = cfg.pop("defaults", [])
    result = OmegaConf.create()
    for item in defaults:
        if not (hasattr(item, 'items') and hasattr(item, '__len__')) or len(item) != 1:
            raise ValueError("Defaults entries must be dictionaries with a single key/value pair.")
        key, value = next(iter(item.items()))
        nested = OmegaConf.load(config_path.parent / key / f"{value}.yaml")
        result = OmegaConf.merge(result, OmegaConf.create({key: nested}))
    result = OmegaConf.merge(result, cfg)
    return OmegaConf.to_container(result, resolve=True)  # type: ignore[return-value]


def load_config(path: str | Path) -> Config:
    config_path = Path(path)
    data = _compose_from_defaults(config_path)
    dataset = DatasetConfig(**data["dataset"])  # type: ignore[arg-type]
    model = ModelConfig(**data["model"])  # type: ignore[arg-type]
    network = NetworkConfig(**data["network"])  # type: ignore[arg-type]
    training = TrainingConfig(**data.get("training", {}))
    callbacks = CallbackConfig(**data.get("callbacks", {}))
    return Config(dataset=dataset, model=model, network=network, training=training, callbacks=callbacks)
