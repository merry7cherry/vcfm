import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp as func_jvp


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


class SinusoidalEmbedding(nn.Module):
    """Standard sinusoidal embedding used for time representations."""

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive for SinusoidalEmbedding")
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim > 1:
            t = t.reshape(t.shape[0], -1)[:, 0]
        if t.ndim == 0:
            t = t.unsqueeze(0)
        half_dim = self.dim // 2
        if half_dim == 0:
            return t[:, None]
        device = t.device
        dtype = t.dtype
        exponent = torch.arange(half_dim, device=device, dtype=dtype)
        exponent = -math.log(self.max_period) * exponent / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class LatentEncoder(nn.Module):
    """Lightweight convolutional encoder that produces a latent posterior."""

    def __init__(
        self,
        *,
        in_channels: int,
        latent_dim: int,
        hidden_channels: int,
        num_layers: int,
        time_embedding_dim: int,
        time_embedding_max_period: float,
        mlp_hidden_dim: int,
        mlp_output_dim: int,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive")
        if mlp_hidden_dim <= 0 or mlp_output_dim <= 0:
            raise ValueError("MLP dimensions must be positive")
        self.time_embedding = SinusoidalEmbedding(
            time_embedding_dim, max_period=time_embedding_max_period
        )
        input_channels = in_channels * 3 + time_embedding_dim
        layers = []
        channels = input_channels
        for _ in range(num_layers):
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
        self.latent_dim = latent_dim
        projection_dim = hidden_channels
        self.mlp = nn.Sequential(
            nn.Linear(projection_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, mlp_output_dim),
            nn.SiLU(),
        )
        self.fc_mu = nn.Linear(mlp_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(mlp_output_dim, latent_dim)

    def forward(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, *spatial = x_0.shape
        t = t.to(device=x_0.device, dtype=x_0.dtype)
        time_emb = self.time_embedding(t).to(dtype=x_0.dtype)
        time_emb = time_emb.view(batch, -1, *([1] * len(spatial)))
        time_emb = time_emb.expand(-1, -1, *spatial)
        inputs = torch.cat([x_0, x_1, x_t, time_emb], dim=1)
        hidden = self.encoder(inputs)
        hidden = self.pool(hidden).flatten(1)
        hidden = self.mlp(hidden)
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        t = _time_broadcast(x_1.shape, device, x_1.dtype)
        x_t = (1 - t) * x_0 + t * x_1

        mu_z, logvar_z = self.latent_encoder(
            x_0.detach(), x_1.detach(), x_t.detach(), t.detach()
        )
        std_z = torch.exp(0.5 * logvar_z)
        eps_z = torch.randn_like(mu_z)
        z = mu_z + std_z * eps_z
        z = z.requires_grad_(True)

        labels_detached = class_labels.detach() if class_labels is not None else None

        def _total_time_derivative(
            fn,
            inputs: Tuple[torch.Tensor, ...],
            tangents: Tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            _, derivative = func_jvp(
                fn,
                inputs,
                tangents,
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

        def _latent_phi(detach_params: bool):
            def wrapped(
                x0_in: torch.Tensor,
                x1_in: torch.Tensor,
                xt_in: torch.Tensor,
                t_in: torch.Tensor,
            ) -> torch.Tensor:
                if detach_params:
                    params = OrderedDict(
                        (name, param.detach().clone())
                        for name, param in self.latent_encoder.named_parameters()
                    )
                    buffers = OrderedDict(
                        (name, buf.detach().clone())
                        for name, buf in self.latent_encoder.named_buffers()
                    )
                else:
                    params = OrderedDict(self.latent_encoder.named_parameters())
                    buffers = OrderedDict(self.latent_encoder.named_buffers())
                args = (x0_in, x1_in, xt_in, t_in)
                mu, logvar = functional_call(self.latent_encoder, (params, buffers), args)
                std = torch.exp(0.5 * logvar)
                return mu + std * eps_z

            return wrapped

        # Theta (velocity network) objectives -------------------------------------------------
        base_velocity = (x_1 - x_0).detach()

        dzdt = _total_time_derivative(
            _latent_phi(detach_params=True),
            (x_0, x_1, x_t, t),
            (
                torch.zeros_like(x_0),
                torch.zeros_like(x_1),
                base_velocity,
                torch.ones_like(t),
            ),
        )

        dudt = _total_time_derivative(
            _velocity_theta(detach_params=True),
            (x_t, t, z),
            (
                base_velocity,
                torch.ones_like(t),
                dzdt,
            ),
        )

        target_velocity = (base_velocity + (1 - t) * dudt).detach()

        fm_residual = self.velocity(
            x_t, t, labels_detached, z
        ) - target_velocity
        fm_loss = fm_residual.reshape(batch, -1).pow(2).mean(dim=1).mean()

        kl_phi_loss = -0.5 * (1 + logvar_z - mu_z.pow(2) - logvar_z.exp())
        kl_phi_loss = kl_phi_loss.sum(dim=1).mean()

        flow_matching_weighted = self.flow_matching_theta_weight * fm_loss
        phi_kl_weighted = self.kl_phi_weight * kl_phi_loss

        total_loss = flow_matching_weighted + phi_kl_weighted

        log_dict = {
            "flow_matching_theta_loss": fm_loss.detach(),
            "flow_matching_weighted_loss": flow_matching_weighted.detach(),
            "kl_phi_loss": kl_phi_loss.detach(),
            "kl_phi_weighted_loss": phi_kl_weighted.detach(),
            "total_loss": total_loss.detach(),
        }

        return total_loss, log_dict

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

