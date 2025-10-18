import copy
from typing import Any, Optional, Tuple

import lightning as L
import torch
from omegaconf import DictConfig

from utils.utils import power_function_beta


class LightningVCFM(L.LightningModule):
    def __init__(self, cfg: DictConfig, model) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.cfg = cfg
        self.model = model

        self.velocity_lr = cfg.model.velocity_learning_rate
        self.coupling_lr = cfg.model.coupling_learning_rate
        self.velocity_weight_decay = cfg.model.velocity_weight_decay
        self.coupling_weight_decay = cfg.model.coupling_weight_decay
        self.use_ema = cfg.model.use_ema
        self.ema_rate = cfg.model.ema_rate
        self.ema_type = cfg.model.ema_type
        self.automatic_optimization = False

        if self.use_ema:
            self.ema = copy.deepcopy(self.model).eval().requires_grad_(False)

        self.log_on_epoch = cfg.log_on_epoch
        self.log_on_step = not cfg.log_on_epoch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            labels = batch[1] if len(batch) > 1 else None
        elif isinstance(batch, dict):
            if 'image' in batch:
                inputs = batch['image']
            elif 'data' in batch:
                inputs = batch['data']
            else:
                raise ValueError("Unsupported batch dictionary keys; expected 'image' or 'data'.")
            labels = batch.get('label')
        else:
            inputs = batch
            labels = None
        if not torch.is_tensor(inputs):
            raise TypeError(f"Expected tensor inputs but received {type(inputs).__name__}.")
        return inputs, labels

    def _format_class_labels(
        self,
        labels: Optional[torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_missing: bool = False,
    ) -> Optional[torch.Tensor]:
        if self.model.label_dim == 0:
            return None
        if labels is None:
            if allow_missing:
                return torch.zeros(batch_size, self.model.label_dim, device=device, dtype=dtype)
            raise RuntimeError("Class-conditional training requires labels from the datamodule.")
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        labels = labels.to(device=device)
        if labels.ndim == 2 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)
        if labels.ndim == 1:
            labels = torch.nn.functional.one_hot(labels.to(torch.int64), num_classes=self.model.label_dim)
        elif labels.ndim == 2 and labels.shape[-1] == self.model.label_dim:
            pass
        else:
            raise ValueError(
                f"Expected labels with shape [batch] or [batch, {self.model.label_dim}] but received {tuple(labels.shape)}."
            )
        if labels.shape[0] != batch_size:
            raise ValueError(
                f"Mismatched label batch dimension: expected {batch_size}, received {labels.shape[0]}."
            )
        labels = labels.to(dtype=dtype)
        return labels

    # ------------------------------------------------------------------
    # Optimizers
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        opt_theta = torch.optim.AdamW(
            self.model.velocity_parameters(),
            lr=self.velocity_lr,
            betas=(0.9, 0.99),
            weight_decay=self.velocity_weight_decay,
        )
        opt_phi = torch.optim.AdamW(
            self.model.coupling_parameters(),
            lr=self.coupling_lr,
            betas=(0.9, 0.99),
            weight_decay=self.coupling_weight_decay,
        )
        return [opt_theta, opt_phi]

    # ------------------------------------------------------------------
    # EMA
    # ------------------------------------------------------------------
    @torch.no_grad()
    def ema_update(self) -> None:
        assert self.use_ema
        ema_rate = self.ema_rate
        if self.ema_type == 'power':
            ema_rate = 1 - power_function_beta(std=self.ema_rate, t=self.global_step)
        for p_ema, p_net in zip(self.ema.parameters(), self.model.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_rate))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def training_step(self, batch: Any, batch_idx: int) -> Optional[torch.Tensor]:
        inputs, raw_labels = self._unpack_batch(batch)
        class_labels = self._format_class_labels(
            raw_labels,
            batch_size=inputs.shape[0],
            device=inputs.device,
            dtype=inputs.dtype,
            allow_missing=False,
        )

        opt_theta, opt_phi = self.optimizers()

        opt_theta.zero_grad(set_to_none=True)
        fm_loss, phi_loss, log_dict = self.model.losses(inputs, class_labels=class_labels)
        self.manual_backward(fm_loss, retain_graph=True)
        opt_theta.step()

        opt_phi.zero_grad(set_to_none=True)
        self.manual_backward(phi_loss)
        opt_phi.step()

        if self.use_ema:
            self.ema_update()

        for key, value in log_dict.items():
            self.log(
                key,
                value,
                on_step=self.log_on_step,
                on_epoch=self.log_on_epoch,
                prog_bar=key == 'flow_matching_loss',
                logger=True,
            )
        self.log(
            'train_loss',
            fm_loss.detach() + phi_loss.detach(),
            on_step=self.log_on_step,
            on_epoch=self.log_on_epoch,
            prog_bar=True,
            logger=True,
        )
        return None

    # ------------------------------------------------------------------
    # Sampling API for callbacks
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        sample_shape,
        n_iters: int,
        use_ema: bool = True,
        class_labels: Optional[torch.Tensor] = None,
    ):
        if use_ema:
            assert self.use_ema
            model = self.ema.eval()
        else:
            model = self.model
        param_source = self.ema if use_ema and self.use_ema else self.model
        dtype = next(param_source.parameters()).dtype
        prepared_labels = self._format_class_labels(
            class_labels,
            batch_size=sample_shape[0],
            device=self.device,
            dtype=dtype,
            allow_missing=True,
        )
        return model.sample(sample_shape, n_iters, self.device, class_labels=prepared_labels)
