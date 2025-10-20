from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Union

import torch

from omegaconf import OmegaConf

from vcfm import DataBundle, Trainer, build_dataloaders, build_model, load_config
from vcfm.config import DatasetConfig
from vcfm.callbacks import Callback, FIDCallback, ImageSamplerCallback


def _make_callbacks(cfg, data: DataBundle, output_dir: Path) -> List[Callback]:
    callbacks = []
    sample_dir = output_dir / "samples"
    if cfg.callbacks.generate and cfg.training.sample_every > 0:
        callbacks.append(
            ImageSamplerCallback(
                sample_shape=cfg.dataset.sample_shape,
                n_iters=cfg.training.sample_steps,
                every=cfg.training.sample_every,
                output_dir=sample_dir,
                use_ema=cfg.model.use_ema,
                plot_type=cfg.dataset.plot_type,
            )
        )
    if cfg.callbacks.fid and data.fid is not None and cfg.training.eval_every > 0:
        callbacks.append(
            FIDCallback(
                real_loader=data.fid,
                sample_shape=cfg.dataset.fid_sample_shape,
                n_iters=cfg.training.sample_steps,
                every=cfg.training.eval_every,
                n_samples=cfg.dataset.n_dataset_samples,
                use_ema=cfg.model.use_ema,
            )
        )
    return callbacks


def _print_config(cfg) -> None:
    print("=== Configuration Parameters ===")
    config_dict = asdict(cfg)

    def _print_nested(data, indent: int = 0) -> None:
        indent_str = " " * indent
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    print(f"{indent_str}{key}:")
                    _print_nested(value, indent + 2)
                else:
                    print(f"{indent_str}{key}: {value}")
        else:
            print(f"{indent_str}{data}")

    for section, params in config_dict.items():
        print(f"{section.upper()}:")
        if isinstance(params, dict):
            _print_nested(params, indent=2)
        else:
            _print_nested(params, indent=2)
        print()


def _format_hparam(value: Union[int, float]) -> str:
    if isinstance(value, float):
        formatted = f"{value:.10g}"
    else:
        formatted = str(value)
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    sanitized = formatted.replace(".", "")
    return sanitized or "0"


def main() -> None:
    parser = argparse.ArgumentParser(description="Variationally-Coupled Flow Matching trainer")
    parser.add_argument("--config", type=str, default="conf/config.yaml", help="Path to the configuration file")
    parser.add_argument("--device", type=str, default="auto", help="Training device: 'auto', 'cpu', or CUDA id")
    parser.add_argument("--output", type=str, default="", help="Directory for logs and samples (overrides config)")
    parser.add_argument("--save", type=str, default="", help="Optional path to store a final checkpoint")
    parser.add_argument(
        "--resume", type=str, default="", help="Optional path to a checkpoint to resume training"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory to store intermediate checkpoints (defaults to output/checkpoints)",
    )
    parser.add_argument(
        "--coupling-num-blocks",
        type=int,
        default=1,
        help="Number of residual blocks for the Gaussian coupling network (default: 1)",
    )
    parser.add_argument("--dataset", type=str, default="", help="Dataset configuration to load (overrides config)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training dataloader (overrides dataset config)",
    )
    parser.add_argument(
        "--straightness-weight",
        type=float,
        default=None,
        help="Straightness loss weight (overrides model config)",
    )
    parser.add_argument(
        "--kl-weight",
        type=float,
        default=None,
        help="KL loss weight (overrides model config)",
    )
    parser.add_argument(
        "--ema-rate",
        type=float,
        default=None,
        help="EMA decay rate (overrides model config)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    config_dir = Path(args.config).resolve().parent

    if args.dataset:
        dataset_key = args.dataset.lower()
        dataset_config_path = config_dir / "dataset" / f"{dataset_key}.yaml"
        if not dataset_config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
        dataset_data = OmegaConf.to_container(OmegaConf.load(dataset_config_path), resolve=True)
        cfg.dataset = DatasetConfig(**dataset_data)  # type: ignore[arg-type]

    if args.batch_size is not None:
        cfg.dataset.batch_size = args.batch_size
    if args.straightness_weight is not None:
        cfg.model.straightness_weight = args.straightness_weight
    if args.kl_weight is not None:
        cfg.model.kl_weight = args.kl_weight
    if args.ema_rate is not None:
        cfg.model.ema_rate = args.ema_rate

    dataset_name = cfg.dataset.name.lower()
    batch_size = cfg.dataset.batch_size
    run_name = "{}_b{}_kl_{}_st_{}_ema_{}".format(
        dataset_name,
        batch_size,
        _format_hparam(cfg.model.kl_weight),
        _format_hparam(cfg.model.straightness_weight),
        _format_hparam(cfg.model.ema_rate),
    )

    output_base = Path(args.output) if args.output else Path(cfg.training.output_dir)
    output_dir = output_base / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    _print_config(cfg)

    data = build_dataloaders(cfg.dataset)
    model = build_model(cfg, coupling_num_blocks=args.coupling_num_blocks)
    callbacks = _make_callbacks(cfg, data, output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    if cfg.training.checkpoint_every > 0:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model,
        cfg,
        data,
        device=device,
        callbacks=callbacks,
        checkpoint_dir=checkpoint_dir if cfg.training.checkpoint_every > 0 else None,
        checkpoint_prefix=run_name,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()

    if args.save:
        trainer.save_checkpoint(args.save)


if __name__ == "__main__":
    main()
