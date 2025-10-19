from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm

from .callbacks import Callback
from .config import Config
from .data import DataBundle
from .utils import power_function_beta, set_seed


def _prepare_batch(batch, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if isinstance(batch, (list, tuple)):
        data = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    elif isinstance(batch, dict):
        data = batch.get("image") or batch.get("data")
        labels = batch.get("label")
    else:
        data = batch
        labels = None
    if data is None:
        raise ValueError("Batch does not contain data tensor")
    x = torch.as_tensor(data, device=device)
    if not torch.is_floating_point(x):
        x = x.float() / 255.0
    if labels is None:
        return x, None
    labels_tensor = torch.as_tensor(labels, device=device)
    return x, labels_tensor


class EMA:
    def __init__(self, model: nn.Module, decay: float, ema_type: str) -> None:
        self.decay = decay
        self.ema_type = ema_type
        self.model = copy.deepcopy(model).eval()

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        decay = self.decay
        if self.ema_type == "power":
            decay = 1.0 - power_function_beta(self.decay, step)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        model_buffers = dict(model.named_buffers())
        for name, ema_buf in self.model.named_buffers():
            if name in model_buffers:
                ema_buf.copy_(model_buffers[name])


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        data: DataBundle,
        *,
        device: Optional[torch.device] = None,
        callbacks: Optional[List[Callback]] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.data = data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callbacks = callbacks or []
        self.global_step = 0
        self.history: Dict[str, List[Tuple[int, float]]] = {}

        set_seed(cfg.training.seed)
        self.model.to(self.device)

        self.use_ema = cfg.model.use_ema
        self.ema: Optional[EMA] = None
        if self.use_ema:
            self.ema = EMA(self.model, cfg.model.ema_rate, cfg.model.ema_type)
            self.ema.to(self.device)

        self.opt_theta = AdamW(
            self.model.velocity_parameters(),
            lr=cfg.model.velocity_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=cfg.model.velocity_weight_decay,
        )
        self.opt_phi = AdamW(
            self.model.coupling_parameters(),
            lr=cfg.model.coupling_learning_rate,
            betas=(0.9, 0.99),
            weight_decay=cfg.model.coupling_weight_decay,
        )

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        current_step = self.global_step if step is None else step
        self.history.setdefault(name, []).append((current_step, float(value)))

    def state_dict(self) -> Dict[str, object]:
        from dataclasses import asdict

        return {
            "model": self.model.state_dict(),
            "ema": self.ema.model.state_dict() if self.ema is not None else None,
            "global_step": self.global_step,
            "config": asdict(self.cfg),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self,
        sample_shape: Tuple[int, ...],
        n_iters: int,
        *,
        use_ema: bool = True,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        model = self.ema.model if (use_ema and self.ema is not None) else self.model
        return model.sample(sample_shape, n_iters, self.device, class_labels=class_labels)

    def fit(self) -> None:
        train_loader = self.data.train
        iterator = iter(train_loader)
        total_steps = self.cfg.model.total_training_steps
        log_every = self.cfg.training.log_every
        eval_every = self.cfg.training.eval_every
        sample_every = self.cfg.training.sample_every
        grad_clip = self.cfg.training.grad_clip

        for callback in self.callbacks:
            callback.on_train_start(self)

        progress = tqdm(range(total_steps), desc="Training", dynamic_ncols=True)
        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            inputs, raw_labels = _prepare_batch(batch, self.device)
            class_labels = self._format_labels(raw_labels, inputs)

            self.opt_theta.zero_grad(set_to_none=True)
            fm_loss, phi_loss, logs = self.model.losses(inputs, class_labels=class_labels)
            fm_loss.backward(retain_graph=True)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.velocity_parameters(), grad_clip)
            self.opt_theta.step()

            self.opt_phi.zero_grad(set_to_none=True)
            phi_loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.coupling_parameters(), grad_clip)
            self.opt_phi.step()

            if self.ema is not None:
                self.ema.update(self.model, self.global_step + 1)

            logs = {key: float(value) for key, value in logs.items()}
            logs["fm_loss"] = float(fm_loss.detach())
            logs["phi_loss"] = float(phi_loss.detach())
            progress.set_postfix(
                {
                    "fm": logs.get("flow_matching_loss", logs.get("fm_loss", 0.0)),
                    "phi": logs.get("phi_loss", 0.0),
                }
            )

            for key, value in logs.items():
                self.log_scalar(key, value)

            for callback in self.callbacks:
                callback.on_step_end(self, self.global_step + 1, logs)

            self.global_step += 1

            if sample_every > 0 and self.global_step % sample_every == 0:
                for callback in self.callbacks:
                    callback.on_sample(self, self.global_step)

            if eval_every > 0 and self.global_step % eval_every == 0:
                for callback in self.callbacks:
                    callback.on_evaluation(self, self.global_step)

        for callback in self.callbacks:
            callback.on_train_end(self)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _format_labels(
        self, labels: Optional[torch.Tensor], inputs: torch.Tensor
    ) -> Optional[torch.Tensor]:
        if not self.cfg.model.class_conditional:
            return None
        if labels is None:
            raise RuntimeError("Class-conditional training requires dataset labels.")
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if labels.ndim == 1:
            labels = torch.nn.functional.one_hot(
                labels.to(torch.int64), num_classes=self.data.num_classes
            )
        elif labels.ndim == 2 and labels.shape[-1] == self.data.num_classes:
            pass
        else:
            raise ValueError(
                "Labels must be either 1D integer class indices or precomputed one-hot vectors."
            )
        labels = labels.to(device=self.device, dtype=inputs.dtype)
        return labels
