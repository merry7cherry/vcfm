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
        checkpoint_dir: Optional[Path] = None,
        checkpoint_prefix: Optional[str] = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.data = data
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.callbacks = callbacks or []
        self.global_step = 0
        self.history: Dict[str, List[Tuple[int, float]]] = {}
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.checkpoint_prefix = checkpoint_prefix or "checkpoint"

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
    @torch.no_grad()
    def _report_gradients(
        self,
        name: str,
        named_parameters: List[Tuple[str, nn.Parameter]],
    ) -> None:
        grads = [param.grad.detach().flatten() for _, param in named_parameters if param.grad is not None]
        if not grads:
            print(f"[Gradients][{name}] No gradients available.")
            return

        flat_grads = torch.cat(grads)
        grad_mean = flat_grads.mean().item()
        grad_std = flat_grads.std(unbiased=False).item()
        grad_min = flat_grads.min().item()
        grad_max = flat_grads.max().item()
        grad_norm = torch.linalg.vector_norm(flat_grads).item()

        print(
            f"[Gradients][{name}] stats -> mean: {grad_mean:.6e}, std: {grad_std:.6e}, "
            f"min: {grad_min:.6e}, max: {grad_max:.6e}, l2-norm: {grad_norm:.6e}"
        )

        for param_name, param in named_parameters:
            if param.grad is None:
                continue
            grad_matrix = param.grad.detach().cpu()
            print(f"[Gradients][{name}] tensor: {param_name}, shape: {tuple(grad_matrix.shape)}")
            print(grad_matrix)
            break

    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        current_step = self.global_step if step is None else step
        self.history.setdefault(name, []).append((current_step, float(value)))

    def state_dict(self) -> Dict[str, object]:
        from dataclasses import asdict

        return {
            "model": self.model.state_dict(),
            "ema": self.ema.model.state_dict() if self.ema is not None else None,
            "global_step": self.global_step,
            "opt_theta": self.opt_theta.state_dict(),
            "opt_phi": self.opt_phi.state_dict(),
            "config": asdict(self.cfg),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), checkpoint_path)

    def load_checkpoint(self, path: str | Path, *, strict: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        model_state = checkpoint.get("model")
        if model_state is None:
            raise KeyError("Checkpoint is missing the 'model' state dictionary.")
        self.model.load_state_dict(model_state, strict=strict)

        opt_theta_state = checkpoint.get("opt_theta")
        if opt_theta_state is not None:
            self.opt_theta.load_state_dict(opt_theta_state)

        opt_phi_state = checkpoint.get("opt_phi")
        if opt_phi_state is not None:
            self.opt_phi.load_state_dict(opt_phi_state)

        ema_state = checkpoint.get("ema")
        if ema_state is not None and self.use_ema:
            if self.ema is None:
                self.ema = EMA(self.model, self.cfg.model.ema_rate, self.cfg.model.ema_type)
                self.ema.to(self.device)
            self.ema.model.load_state_dict(ema_state, strict=strict)

        self.global_step = int(checkpoint.get("global_step", 0))

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

        progress = tqdm(
            range(self.global_step, total_steps),
            desc="Training",
            dynamic_ncols=True,
            initial=min(self.global_step, total_steps),
            total=total_steps,
        )
        for _ in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)

            inputs, raw_labels = _prepare_batch(batch, self.device)
            class_labels = self._format_labels(raw_labels, inputs)

            self.opt_theta.zero_grad(set_to_none=True)
            theta_loss, phi_loss, logs = self.model.losses(inputs, class_labels=class_labels)
            theta_loss.backward(retain_graph=True)
            velocity_params = list(self.model.velocity_net.named_parameters())
            self._report_gradients("theta", velocity_params)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.velocity_parameters(), grad_clip)
            self.opt_theta.step()

            self.opt_phi.zero_grad(set_to_none=True)
            phi_loss.backward()
            coupling_params = list(self.model.coupling_net.named_parameters())
            self._report_gradients("phi", coupling_params)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.coupling_parameters(), grad_clip)
            self.opt_phi.step()

            if self.ema is not None:
                self.ema.update(self.model, self.global_step + 1)

            logs = {key: float(value) for key, value in logs.items()}
            logs["theta_loss"] = float(theta_loss.detach())
            logs["phi_loss"] = float(phi_loss.detach())
            progress.set_postfix(
                {
                    "fmθ": logs.get("flow_matching_theta_loss", logs.get("theta_loss", 0.0)),
                    "strθ": logs.get("straightness_theta_loss", 0.0),
                    "strφ": logs.get("straightness_phi_loss", 0.0),
                    "klφ": logs.get("kl_phi_loss", 0.0),
                }
            )

            for key, value in logs.items():
                self.log_scalar(key, value)

            for callback in self.callbacks:
                callback.on_step_end(self, self.global_step + 1, logs)

            self.global_step += 1

            checkpoint_every = self.cfg.training.checkpoint_every
            if (
                checkpoint_every > 0
                and self.checkpoint_dir is not None
                and self.global_step % checkpoint_every == 0
            ):
                checkpoint_name = f"{self.checkpoint_prefix}_step-{self.global_step:07d}.pt"
                self.save_checkpoint(self.checkpoint_dir / checkpoint_name)

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
