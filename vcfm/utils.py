from __future__ import annotations

import random
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader


__all__ = [
    "set_seed",
    "rescaling_inv",
    "adjust_channels",
    "power_function_beta",
    "ResumableDataLoader",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rescaling_inv(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * x + 0.5


def adjust_channels(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW tensor but received shape {tuple(images.shape)}")
    channels = images.size(1)
    if channels == 1:
        return images.repeat(1, 3, 1, 1)
    if channels == 3:
        return images
    raise ValueError(f"Unexpected number of channels: {channels}. Expected 1 or 3.")


def _std_to_exp(std: float | torch.Tensor | np.ndarray) -> np.ndarray:
    std_np = np.asarray(std, dtype=np.float64)
    tmp = std_np.flatten() ** -2
    exp = [np.roots([1, 7, 16 - t, 12 - t]).real.max() for t in tmp]
    exp = np.float64(exp).reshape(std_np.shape)
    return exp


def power_function_beta(std: float, t: int) -> float:
    exp = _std_to_exp(std)
    beta = (1 - (1 / (t + 1))) ** (exp + 1)
    return float(beta)


class ResumableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.iteration = 0

    def __iter__(self) -> Iterator:
        for batch in super().__iter__():
            self.iteration += 1
            yield batch

    def state_dict(self) -> dict:
        return {"iteration": self.iteration}

    def load_state_dict(self, state_dict: dict) -> None:
        self.iteration = state_dict.get("iteration", 0)
