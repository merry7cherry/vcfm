import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jvp as autograd_jvp
from torch.func import functional_call


def _time_broadcast(
    shape: torch.Size, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.rand((shape[0],) + (1,) * (len(shape) - 1), device=device, dtype=dtype)


def _prepare_class_labels(
    class_labels: Optional[torch.Tensor],
    *,
    batch_size: int,
    label_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Validate, format, and broadcast class labels."""

    if label_dim == 0:
        return None
    if class_labels is None:
        raise ValueError(
            "Class labels must be provided when using a class-conditional model."
        )
    if class_labels.ndim == 0:
        class_labels = class_labels.unsqueeze(0)
    if class_labels.ndim == 1:
        class_labels = F.one_hot(class_labels.to(torch.int64), num_classes=label_dim)
    elif class_labels.ndim != 2 or class_labels.shape[-1] != label_dim:
        raise ValueError(
            "Class labels must be 1D indices or 2D one-hot vectors matching label_dim."
        )
    if class_labels.shape[0] not in {1, batch_size}:
        raise ValueError(
            "Class labels must match the batch size or provide a single label to broadcast."
        )
    if class_labels.shape[0] == 1 and batch_size > 1:
        class_labels = class_labels.expand(batch_size, -1)
    return class_labels.to(device=device, dtype=dtype)


class LatentEncoder(nn.Module):
    """Lightweight convolutional encoder that produces a latent posterior."""

    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        hidden_channels: int,
        num_layers: int,
        label_dim: int = 0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        input_channels = in_channels * 3 + 1
        layers = []
        channels = input_channels
        for layer_idx in range(num_layers):
            layers.append(
                nn.Conv2d(
                    channels,
                    hidden_channels,
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(
                nn.GroupNorm(
                    num_groups=min(32, hidden_channels), num_channels=hidden_channels
                )
            )
            layers.append(nn.SiLU())
            channels = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        projection_dim = hidden_channels
        if label_dim > 0:
            self.label_embed = nn.Linear(label_dim, projection_dim)
        else:
            self.label_embed = None
        self.fc_mu = nn.Linear(projection_dim, latent_dim)
        self.fc_logvar = nn.Linear(projection_dim, latent_dim)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat([x_0, x_1, x_t, t], dim=1)
        hidden = self.encoder(inputs)
        hidden = self.pool(hidden).flatten(1)
        if self.label_embed is not None:
            if class_labels is None:
                raise ValueError(
                    "Class labels must be provided for conditional latent encoding."
                )
            hidden = hidden + self.label_embed(class_labels)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar


class VariationallyCoupledFlowMatching(nn.Module):
    """Variationally-Coupled Flow Matching with latent-conditioning."""

    def __init__(
        self,
        velocity_net: nn.Module,
        latent_encoder: LatentEncoder,
        *,
        sigma_min: float,
        sigma_max: float,
        flow_matching_theta_weight: float,
        straightness_weight: float,
        kl_phi_weight: float,
        label_dim: int,
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.velocity_net = velocity_net
        self.latent_encoder = latent_encoder
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.flow_matching_theta_weight = flow_matching_theta_weight
        self.straightness_weight = straightness_weight
        self.kl_phi_weight = kl_phi_weight
        self.label_dim = label_dim
        self.latent_dim = latent_dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def velocity_parameters(self):
        return self.velocity_net.parameters()

    def coupling_parameters(self):
        return self.latent_encoder.parameters()

    def _flatten_time(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            return t
        return t.reshape(t.shape[0], -1)[:, 0]

    def _time_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        t_flat = self._flatten_time(t)
        log_sigma_min = math.log(self.sigma_min)
        log_sigma_max = math.log(self.sigma_max)
        log_sigma = (1 - t_flat) * log_sigma_max + t_flat * log_sigma_min
        return torch.exp(log_sigma)

    def _noise_labels(self, t: torch.Tensor) -> torch.Tensor:
        sigma = self._time_to_sigma(t)
        return torch.log(sigma) / 4.0

    def _velocity_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        z: torch.Tensor,
        *,
        detach_params: bool,
    ) -> torch.Tensor:
        noise_labels = self._noise_labels(t)
        if detach_params:
            params = OrderedDict(
                (name, param.detach().clone())
                for name, param in self.velocity_net.named_parameters()
            )
            buffers = OrderedDict(
                (name, buf.detach().clone())
                for name, buf in self.velocity_net.named_buffers()
            )
        else:
            params = OrderedDict(self.velocity_net.named_parameters())
            buffers = OrderedDict(self.velocity_net.named_buffers())
        args = (x, noise_labels, class_labels, z)
        return functional_call(self.velocity_net, (params, buffers), args)

    def velocity(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_labels: Optional[torch.Tensor],
        z: torch.Tensor,
    ) -> torch.Tensor:
        return self._velocity_forward(x, t, class_labels, z, detach_params=False)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def losses(
        self, x_1: torch.Tensor, *, class_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        device = x_1.device
        batch = x_1.shape[0]

        class_labels = _prepare_class_labels(
            class_labels,
            batch_size=batch,
            label_dim=self.label_dim,
            device=device,
            dtype=x_1.dtype,
        )

        x_0 = torch.randn_like(x_1)
        x_0 = x_0.requires_grad_(True)

        t = _time_broadcast(x_1.shape, device, x_1.dtype)
        t = t.requires_grad_(True)
        x_t = (1 - t) * x_0 + t * x_1
        x_t = x_t.requires_grad_(True)
        u = (x_1 - x_0).detach()

        mu_z, logvar_z = self.latent_encoder(
            x_0.detach(), x_1.detach(), x_t.detach(), t.detach(), class_labels
        )
        std_z = torch.exp(0.5 * logvar_z)
        eps_z = torch.randn_like(mu_z)
        z = mu_z + std_z * eps_z
        z = z.requires_grad_(True)

        labels_detached = class_labels.detach() if class_labels is not None else None

        def _total_time_derivative(
            fn,
            inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            tangents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ) -> torch.Tensor:
            _, derivative = autograd_jvp(
                fn,
                inputs,
                tangents,
                create_graph=True,
                strict=True,
            )
            return derivative

        def _velocity_theta(detach_params: bool):
            def wrapped(
                x_in: torch.Tensor, t_in: torch.Tensor, z_in: torch.Tensor
            ) -> torch.Tensor:
                return self._velocity_forward(
                    x_in,
                    t_in,
                    labels_detached,
                    z_in,
                    detach_params=detach_params,
                )

            return wrapped

        # Theta (velocity network) objectives -------------------------------------------------
        x_t_theta = x_t.detach().clone()
        t_theta = t.detach().clone()
        z_theta = z.detach()

        fm_residual = self.velocity(
            x_t_theta, t_theta, labels_detached, z_theta
        ) - u
        fm_loss = fm_residual.reshape(batch, -1).pow(2).mean(dim=1).mean()

        tangent = (
            (x_1 - x_0).detach(),
            torch.ones_like(t),
            torch.zeros_like(z),
        )
        straightness = _total_time_derivative(
            _velocity_theta(detach_params=False),
            (x_t, t, z),
            tangent,
        )
        straightness_loss = (
            straightness.reshape(batch, -1).pow(2).sum(dim=1).mean()
        )

        kl_phi_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        kl_phi_loss = kl_phi_loss.sum(dim=1).mean()

        theta_components = {"flow_matching_theta_loss": fm_loss}
        phi_components = {"kl_phi_loss": kl_phi_loss}

        straightness_weighted = self.straightness_weight * straightness_loss

        theta_loss = (
            self.flow_matching_theta_weight * theta_components["flow_matching_theta_loss"]
            + straightness_weighted
        )

        phi_kl_weighted = self.kl_phi_weight * phi_components["kl_phi_loss"]
        phi_total_loss = phi_kl_weighted + straightness_weighted
        phi_loss = phi_kl_weighted

        component_logs = {
            **{name: value.detach() for name, value in theta_components.items()},
            **{name: value.detach() for name, value in phi_components.items()},
            "straightness_loss": straightness_loss.detach(),
            "straightness_weighted_loss": straightness_weighted.detach(),
            "phi_total_loss": phi_total_loss.detach(),
        }
        log_dict = {
            **component_logs,
            "theta_loss": theta_loss.detach(),
            "phi_loss": phi_loss.detach(),
        }

        return theta_loss, phi_loss, log_dict

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        sample_shape: Tuple[int, ...],
        n_iters: int,
        device: torch.device,
        *,
        class_labels: Optional[torch.Tensor] = None,
        z: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dtype = next(self.velocity_net.parameters()).dtype
        batch = sample_shape[0]
        class_labels = _prepare_class_labels(
            class_labels,
            batch_size=batch,
            label_dim=self.label_dim,
            device=device,
            dtype=dtype,
        )
        x = torch.randn(sample_shape, device=device, dtype=dtype)
        if z is None:
            z = torch.randn(batch, self.latent_dim, device=device, dtype=dtype)
        else:
            if z.ndim != 2 or z.shape[-1] != self.latent_dim:
                raise ValueError(
                    "Provided latent codes must have shape (batch, latent_dim)."
                )
            if z.shape[0] not in {1, batch}:
                raise ValueError(
                    "Number of provided latents must be 1 or match the batch size."
                )
            if z.shape[0] == 1 and batch > 1:
                z = z.expand(batch, -1)
            z = z.to(device=device, dtype=dtype)
        if class_labels is not None and self.label_dim > 0:
            class_labels = class_labels.to(device=device, dtype=dtype)
        dt = 1.0 / max(n_iters, 1)
        for step in range(n_iters):
            t_value = torch.full(
                (sample_shape[0],) + (1,) * (len(sample_shape) - 1),
                dt * step,
                device=device,
                dtype=dtype,
            )
            v = self.velocity(x, t_value, class_labels, z)
            x = x + dt * v
        return x

