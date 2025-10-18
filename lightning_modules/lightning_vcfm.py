import copy
from typing import Any, Optional

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
        if isinstance(batch, list):
            inputs = batch[0]
            labels = batch[1]
            if self.model.label_dim > 0:
                if self.cfg.dataset.name != 'imagenet':
                    labels = torch.nn.functional.one_hot(
                        labels.to(torch.int64), num_classes=self.model.label_dim
                    )
                labels = labels.to(device=inputs.device)
                if labels.dtype != inputs.dtype:
                    labels = labels.to(inputs.dtype)
            else:
                labels = None
        else:
            inputs = batch
            labels = None

        opt_theta, opt_phi = self.optimizers()

        opt_theta.zero_grad(set_to_none=True)
        fm_loss, phi_loss, log_dict = self.model.losses(inputs, class_labels=labels)
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
        return model.sample(sample_shape, n_iters, self.device, class_labels=class_labels)
