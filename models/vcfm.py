import math
from collections import OrderedDict
from typing import Dict, Mapping, Optional, Tuple

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


class GaussianCoupling(nn.Module):
    """Gaussian variational coupling q_\phi(x_0 | x_1)."""

    def __init__(
        self,
        net: nn.Module,
        label_dim: int = 0,
        min_log_std: float = -7.0,
        max_log_std: float = 5.0,
    ) -> None:
        super().__init__()
        self.net = net
        self.label_dim = label_dim
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(
        self, x_1: torch.Tensor, class_labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x_1.shape[0]
        device = x_1.device
        dtype = x_1.dtype
        noise_labels = torch.zeros(batch, device=device, dtype=dtype)
        class_labels = _prepare_class_labels(
            class_labels,
            batch_size=batch,
            label_dim=self.label_dim,
            device=device,
            dtype=dtype,
        )
        outputs = self.net(x_1, noise_labels, class_labels)
        mu, log_sigma = torch.chunk(outputs, 2, dim=1)
        log_sigma = torch.clamp(log_sigma, self.min_log_std, self.max_log_std)
        return mu, log_sigma


class VariationallyCoupledFlowMatching(nn.Module):
    """
    Variationally-Coupled Flow Matching (VC-FM).
    """

    def __init__(
        self,
        velocity_net: nn.Module,
        coupling_net: GaussianCoupling,
        *,
        sigma_min: float,
        sigma_max: float,
        flow_matching_theta_weight: float,
        straightness_theta_weight: float,
        straightness_phi_weight: float,
        kl_phi_weight: float,
        label_dim: int,
    ) -> None:
        super().__init__()
        self.velocity_net = velocity_net
        self.coupling_net = coupling_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.flow_matching_theta_weight = flow_matching_theta_weight
        self.straightness_theta_weight = straightness_theta_weight
        self.straightness_phi_weight = straightness_phi_weight
        self.kl_phi_weight = kl_phi_weight
        self.label_dim = label_dim

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def velocity_parameters(self):
        return self.velocity_net.parameters()

    def coupling_parameters(self):
        return self.coupling_net.parameters()

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
        args = (x, noise_labels, class_labels)
        return functional_call(self.velocity_net, (params, buffers), args)

    def velocity(
        self, x: torch.Tensor, t: torch.Tensor, class_labels: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self._velocity_forward(x, t, class_labels, detach_params=False)

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def losses(
        self,
        x_1: torch.Tensor,
        *,
        class_labels: Optional[torch.Tensor] = None,
        phi_loss_weights: Optional[Mapping[str, float]] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        device = x_1.device
        batch = x_1.shape[0]

        class_labels = _prepare_class_labels(
            class_labels,
            batch_size=batch,
            label_dim=self.label_dim,
            device=device,
            dtype=x_1.dtype,
        )

        eps = torch.randn_like(x_1)
        mu, log_sigma = self.coupling_net(x_1, class_labels)
        sigma = torch.exp(log_sigma)
        x_0 = mu + sigma * eps
        x_0 = x_0.requires_grad_(True)

        t = _time_broadcast(x_1.shape, device, x_1.dtype)
        t = t.requires_grad_(True)
        x_t = (1 - t) * x_0 + t * x_1
        x_t = x_t.requires_grad_(True)
        u = (x_1 - x_0).detach()

        labels_detached = class_labels.detach() if class_labels is not None else None

        def _total_time_derivative(
            fn, inputs: Tuple[torch.Tensor, torch.Tensor], tangents: Tuple[torch.Tensor, torch.Tensor]
        ) -> torch.Tensor:
            _, derivative = autograd_jvp(
                fn,
                inputs,
                tangents,
                create_graph=True,
                strict=True,
            )
            return derivative

        def _velocity_fn(detach_params: bool):
            def wrapped(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
                return self._velocity_forward(
                    x_in,
                    t_in,
                    labels_detached,
                    detach_params=detach_params,
                )

            return wrapped

        # Theta (velocity network) objectives -------------------------------------------------
        x_t_theta = x_t.detach().clone().requires_grad_(True)
        t_theta = t.detach().clone().requires_grad_(True)
        tangent_theta = ((x_1 - x_0).detach(), torch.ones_like(t_theta))

        fm_residual = self.velocity(x_t_theta, t_theta, labels_detached) - u
        fm_loss = fm_residual.reshape(batch, -1).pow(2).mean(dim=1).mean()

        total_derivative_theta = _total_time_derivative(
            _velocity_fn(detach_params=False),
            (x_t_theta, t_theta),
            tangent_theta,
        )
        straightness_loss_theta = (
            total_derivative_theta.reshape(batch, -1).pow(2).sum(dim=1).mean()
        )

        theta_components = {
            "flow_matching_theta_loss": fm_loss,
            "straightness_theta_loss": straightness_loss_theta,
        }
        theta_loss = self.flow_matching_theta_weight * theta_components["flow_matching_theta_loss"]

        # Phi (coupling network) objectives ---------------------------------------------------
        tangent_phi = ((x_1 - x_0).detach(), torch.ones_like(t))
        total_derivative_phi = _total_time_derivative(
            _velocity_fn(detach_params=True),
            (x_t, t),
            tangent_phi,
        )
        straightness_loss_phi = (
            total_derivative_phi.reshape(batch, -1).pow(2).sum(dim=1).mean()
        )

        kl_phi_loss = 0.5 * (
            (mu.pow(2) + sigma.pow(2) - 1.0 - 2.0 * log_sigma)
            .reshape(batch, -1)
            .sum(dim=1)
            .mean()
        )

        phi_components = {
            "straightness_phi_loss": straightness_loss_phi,
            "kl_phi_loss": kl_phi_loss,
        }
        if phi_loss_weights is None:
            phi_loss_weights = {
                "straightness_phi_loss": self.straightness_phi_weight,
                "kl_phi_loss": self.kl_phi_weight,
            }
        phi_loss = sum(
            phi_loss_weights.get(name, 0.0) * component
            for name, component in phi_components.items()
        )

        component_logs = {
            **{name: value.detach() for name, value in theta_components.items()},
            **{name: value.detach() for name, value in phi_components.items()},
        }
        log_dict = {
            **component_logs,
            "theta_loss": theta_loss.detach(),
            "phi_loss": phi_loss.detach(),
        }

        return theta_loss, phi_loss, log_dict, theta_components, phi_components

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
            v = self.velocity(x, t_value, class_labels)
            x = x + dt * v
        return x
