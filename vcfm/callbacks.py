from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid, save_image

from .utils import adjust_channels, rescaling_inv


if TYPE_CHECKING:
    from .trainer import Trainer


class Callback:
    def on_train_start(self, trainer: "Trainer") -> None:
        pass

    def on_step_end(self, trainer: "Trainer", step: int, logs: dict) -> None:
        pass

    def on_sample(self, trainer: "Trainer", step: int) -> None:
        pass

    def on_evaluation(self, trainer: "Trainer", step: int) -> None:
        pass

    def on_train_end(self, trainer: "Trainer") -> None:
        pass


class ImageSamplerCallback(Callback):
    def __init__(
        self,
        sample_shape,
        n_iters: int,
        every: int,
        output_dir: Path,
        *,
        use_ema: bool = True,
        plot_type: str = "grid",
        seed: int = 32,
    ) -> None:
        self.sample_shape = tuple(sample_shape)
        self.n_iters = n_iters
        self.every = every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_ema = use_ema
        self.plot_type = plot_type
        self.seed = seed

    def on_train_start(self, trainer: "Trainer") -> None:
        self._generate(trainer, step=0)

    def on_sample(self, trainer: "Trainer", step: int) -> None:
        if self.every > 0 and step % self.every == 0:
            self._generate(trainer, step=step)

    def _generate(self, trainer: "Trainer", step: int) -> None:
        device = trainer.device
        if device.type == "cuda" and device.index is not None:
            fork_devices = [device.index]
        else:
            fork_devices = []
        dtype = next(trainer.model.parameters()).dtype
        labels = self._prepare_labels(trainer, dtype)
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.seed)
            samples = trainer.sample(
                self.sample_shape, self.n_iters, use_ema=self.use_ema, class_labels=labels
            )
        samples = samples.detach().cpu().clamp(-1, 1)
        if self.plot_type == "scatter":
            self._save_scatter(samples, step)
        else:
            grid = make_grid(samples, nrow=int(math.sqrt(self.sample_shape[0])), value_range=(-1, 1))
            save_image(grid, self.output_dir / f"samples_step_{step:07d}.png")

    def _prepare_labels(self, trainer: "Trainer", dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not trainer.cfg.model.class_conditional or trainer.data.num_classes == 0:
            return None
        num = self.sample_shape[0]
        labels = torch.arange(num, device=trainer.device) % trainer.data.num_classes
        one_hot = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=trainer.data.num_classes)
        return one_hot.to(dtype=dtype)

    def _save_scatter(self, samples: torch.Tensor, step: int) -> None:
        import matplotlib.pyplot as plt

        samples_np = samples.view(samples.shape[0], -1).numpy()
        fig, ax = plt.subplots()
        ax.scatter(samples_np[:, 0], samples_np[:, 1], s=1)
        ax.set_title(f"Samples step {step}")
        fig.savefig(self.output_dir / f"samples_step_{step:07d}.png")
        plt.close(fig)


class FIDCallback(Callback):
    def __init__(
        self,
        real_loader,
        sample_shape,
        n_iters: int,
        every: int,
        n_samples: int,
        *,
        use_ema: bool = True,
    ) -> None:
        self.real_loader = real_loader
        self.sample_shape = tuple(sample_shape)
        self.n_iters = n_iters
        self.every = every
        self.n_samples = n_samples
        self.use_ema = use_ema
        self.metric: Optional[FrechetInceptionDistance] = None
        self.best = float("inf")

    def on_train_start(self, trainer: "Trainer") -> None:
        if self.real_loader is None:
            return
        self.metric = FrechetInceptionDistance(reset_real_features=False, normalize=True).to(trainer.device)
        with torch.no_grad():
            for batch in self.real_loader:
                real, _ = batch if isinstance(batch, (list, tuple)) else (batch, None)
                real = real.to(trainer.device)
                real = adjust_channels(real)
                self.metric.update(real.detach(), real=True)

    def on_evaluation(self, trainer: "Trainer", step: int) -> None:
        if self.metric is None or self.every <= 0 or step % self.every != 0:
            return
        metric = self.metric
        metric.reset()
        total = 0
        while total < self.n_samples:
            labels = None
            if trainer.cfg.model.class_conditional and trainer.data.num_classes > 0:
                rand = torch.randint(
                    0, trainer.data.num_classes, (self.sample_shape[0],), device=trainer.device
                )
                labels = torch.nn.functional.one_hot(rand, num_classes=trainer.data.num_classes)
            samples = trainer.sample(
                self.sample_shape, self.n_iters, use_ema=self.use_ema, class_labels=labels
            )
            samples = rescaling_inv(samples.clamp(-1, 1))
            samples = adjust_channels(samples)
            metric.update(samples.detach().to(trainer.device), real=False)
            total += samples.shape[0]
        fid_value = float(metric.compute())
        trainer.log_scalar(f"fid_{self.n_iters}", fid_value, step=step)
        if fid_value < self.best:
            self.best = fid_value
        trainer.log_scalar(f"fid_{self.n_iters}_best", self.best, step=step)


__all__ = [
    "Callback",
    "ImageSamplerCallback",
    "FIDCallback",
]
