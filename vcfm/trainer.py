from __future__ import annotations

import copy
from dataclasses import dataclass
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
    def __init__(self, model: nn.Module, ema_type: str) -> None:
        self.ema_type = ema_type
        self.model = copy.deepcopy(model).eval()

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    @torch.no_grad()
    def update(
        self,
        model: nn.Module,
        step: int,
        *,
        theta_decay: float,
        phi_decay: float,
        default_decay: Optional[float] = None,
    ) -> None:
        def _resolve(decay_value: float) -> float:
            if self.ema_type == "power":
                return 1.0 - power_function_beta(decay_value, step)
            return decay_value

        theta_resolved = _resolve(theta_decay)
        phi_resolved = _resolve(phi_decay)
        fallback_decay = _resolve(
            default_decay if default_decay is not None else max(theta_decay, phi_decay)
        )

        for (ema_name, ema_param), (name, param) in zip(
            self.model.named_parameters(), model.named_parameters()
        ):
            if ema_name != name:
                raise RuntimeError(
                    "EMA model parameter order does not match source model order."
                )
            if name.startswith("velocity_net"):
                decay_value = theta_resolved
            elif name.startswith("coupling_net"):
                decay_value = phi_resolved
            else:
                decay_value = fallback_decay
            ema_param.data.mul_(decay_value).add_(param.data, alpha=1.0 - decay_value)
        model_buffers = dict(model.named_buffers())
        for name, ema_buf in self.model.named_buffers():
            if name in model_buffers:
                ema_buf.copy_(model_buffers[name])


@dataclass(frozen=True)
class TrainingStage:
    name: str
    end_step: Optional[int]
    phi_loss_weights: Dict[str, float]
    theta_ema_decay: float
    phi_ema_decay: float


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
            self.ema = EMA(self.model, cfg.model.ema_type)
            self.ema.to(self.device)

        self.training_stages = self._build_training_stages()
        self._current_stage_index: Optional[int] = None

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
    # Training stage management
    # ------------------------------------------------------------------
    def _build_training_stages(self) -> List[TrainingStage]:
        total_steps = self.cfg.model.total_training_steps
        cumulative = 0

        def add_stage(
            stages: List[TrainingStage],
            *,
            name: str,
            length: int,
            phi_loss_weights: Dict[str, float],
            theta_ema_decay: float,
            phi_ema_decay: float,
        ) -> None:
            nonlocal cumulative
            if length <= 0 or cumulative >= total_steps:
                return
            cumulative = min(cumulative + length, total_steps)
            stages.append(
                TrainingStage(
                    name=name,
                    end_step=cumulative,
                    phi_loss_weights=dict(phi_loss_weights),
                    theta_ema_decay=theta_ema_decay,
                    phi_ema_decay=phi_ema_decay,
                )
            )

        stages: List[TrainingStage] = []
        phi_only_kl = {
            "straightness_phi_loss": 0.0,
            "kl_phi_loss": self.model.kl_phi_weight,
        }
        phi_full = {
            "straightness_phi_loss": self.model.straightness_phi_weight,
            "kl_phi_loss": self.model.kl_phi_weight,
        }

        add_stage(
            stages,
            name="phi_warmup",
            length=self.cfg.training.phi_warmup_iters,
            phi_loss_weights=phi_only_kl,
            theta_ema_decay=0.99,
            phi_ema_decay=0.9,
        )
        add_stage(
            stages,
            name="theta_warmup",
            length=self.cfg.training.theta_warmup_iters,
            phi_loss_weights=phi_only_kl,
            theta_ema_decay=0.9,
            phi_ema_decay=0.99,
        )
        add_stage(
            stages,
            name="early_training",
            length=self.cfg.training.early_phase_iters,
            phi_loss_weights=phi_full,
            theta_ema_decay=0.99,
            phi_ema_decay=0.9999,
        )
        add_stage(
            stages,
            name="late_training",
            length=self.cfg.training.late_phase_iters,
            phi_loss_weights=phi_full,
            theta_ema_decay=0.99,
            phi_ema_decay=0.999,
        )

        stages.append(
            TrainingStage(
                name="final_training",
                end_step=None,
                phi_loss_weights=dict(phi_full),
                theta_ema_decay=0.99,
                phi_ema_decay=0.99,
            )
        )
        return stages

    def _get_stage_for_step(self, step: int) -> Tuple[TrainingStage, int]:
        for index, stage in enumerate(self.training_stages):
            if stage.end_step is None or step < stage.end_step:
                return stage, index
        # Fallback to the final stage
        final_index = len(self.training_stages) - 1
        return self.training_stages[final_index], final_index

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _component_gradients(
        self,
        loss: torch.Tensor,
        named_parameters: List[Tuple[str, nn.Parameter]],
        retain_graph: bool,
    ) -> List[Tuple[str, Optional[torch.Tensor]]]:
        grads = torch.autograd.grad(
            loss,
            [param for _, param in named_parameters],
            retain_graph=retain_graph,
            allow_unused=True,
        )
        return [
            (name, grad.detach() if grad is not None else None)
            for (name, _), grad in zip(named_parameters, grads)
        ]

    @torch.no_grad()
    def _report_gradients(
        self,
        name: str,
        named_parameters: List[Tuple[str, nn.Parameter]],
        gradients: Optional[List[Tuple[str, Optional[torch.Tensor]]]] = None,
    ) -> None:
        if gradients is None:
            grads = [
                param.grad.detach().flatten()
                for _, param in named_parameters
                if param.grad is not None
            ]
        else:
            grads = [grad.detach().flatten() for _, grad in gradients if grad is not None]
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

        if gradients is None:
            source = ((param_name, param.grad) for param_name, param in named_parameters)
        else:
            source = gradients
        for param_name, grad in source:
            if grad is None:
                continue
            grad_matrix = grad.detach().cpu()
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
                self.ema = EMA(self.model, self.cfg.model.ema_type)
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

            stage, stage_index = self._get_stage_for_step(self.global_step)
            if stage_index != self._current_stage_index:
                print(f"[Trainer] Entering stage '{stage.name}' at step {self.global_step}")
                self._current_stage_index = stage_index

            self.opt_theta.zero_grad(set_to_none=True)
            (
                theta_loss,
                phi_loss,
                logs,
                theta_components,
                phi_components,
            ) = self.model.losses(
                inputs,
                class_labels=class_labels,
                phi_loss_weights=stage.phi_loss_weights,
            )
            velocity_params = list(self.model.velocity_net.named_parameters())
            for component_name, component_loss in theta_components.items():
                component_grads = self._component_gradients(
                    component_loss,
                    velocity_params,
                    retain_graph=True,
                )
                self._report_gradients(
                    f"theta[{component_name}]",
                    velocity_params,
                    gradients=component_grads,
                )
            theta_loss.backward(retain_graph=True)
            self._report_gradients("theta", velocity_params)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.velocity_parameters(), grad_clip)
            self.opt_theta.step()

            self.opt_phi.zero_grad(set_to_none=True)
            coupling_params = list(self.model.coupling_net.named_parameters())
            for component_name, component_loss in phi_components.items():
                component_grads = self._component_gradients(
                    component_loss,
                    coupling_params,
                    retain_graph=True,
                )
                self._report_gradients(
                    f"phi[{component_name}]",
                    coupling_params,
                    gradients=component_grads,
                )
            phi_loss.backward()
            self._report_gradients("phi", coupling_params)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.coupling_parameters(), grad_clip)
            self.opt_phi.step()

            if self.ema is not None:
                self.ema.update(
                    self.model,
                    self.global_step + 1,
                    theta_decay=stage.theta_ema_decay,
                    phi_decay=stage.phi_ema_decay,
                )

            logs = {key: float(value) for key, value in logs.items()}
            logs["theta_loss"] = float(theta_loss.detach())
            logs["phi_loss"] = float(phi_loss.detach())
            progress.set_postfix(
                {
                    "fmθ": logs.get("flow_matching_theta_loss", logs.get("theta_loss", 0.0)),
                    "strθ": logs.get("straightness_theta_loss", 0.0),
                    "strφ": logs.get("straightness_phi_loss", 0.0),
                    "klφ": logs.get("kl_phi_loss", 0.0),
                    "stage": stage.name,
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
