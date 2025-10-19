from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch

from vcfm import DataBundle, Trainer, build_dataloaders, build_model, load_config
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
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_path = Path(args.output) if args.output else Path(cfg.training.output_dir)
    output_dir = output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data = build_dataloaders(cfg.dataset)
    model = build_model(cfg, coupling_num_blocks=args.coupling_num_blocks)
    callbacks = _make_callbacks(cfg, data, output_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else output_dir / "checkpoints"
    trainer = Trainer(
        model,
        cfg,
        data,
        device=device,
        callbacks=callbacks,
        checkpoint_dir=checkpoint_dir if cfg.training.checkpoint_every > 0 else None,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.fit()

    if args.save:
        trainer.save_checkpoint(args.save)


if __name__ == "__main__":
    main()
