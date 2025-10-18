from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils.dataset import ImageFolderDataset

from .config import DatasetConfig
from .utils import ResumableDataLoader


@dataclass
class DataBundle:
    train: DataLoader
    fid: Optional[DataLoader]
    num_classes: int


def _build_cifar10(cfg: DatasetConfig) -> DataBundle:
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    fid_transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=train_transform)
    fid_ds = datasets.CIFAR10(cfg.data_dir, train=True, download=True, transform=fid_transform)
    train_loader = ResumableDataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    fid_loader = DataLoader(
        fid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return DataBundle(train=train_loader, fid=fid_loader, num_classes=cfg.label_dim)


def _build_mnist(cfg: DatasetConfig, dataset_cls) -> DataBundle:
    normalize = transforms.Normalize((0.5,), (0.5,))
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])
    fid_transform = transforms.Compose([transforms.ToTensor()])
    train_ds = dataset_cls(cfg.data_dir, train=True, download=True, transform=train_transform)
    fid_ds = dataset_cls(cfg.data_dir, train=True, download=True, transform=fid_transform)
    train_loader = ResumableDataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    fid_loader = DataLoader(
        fid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return DataBundle(train=train_loader, fid=fid_loader, num_classes=cfg.label_dim)


def _build_imagefolder(cfg: DatasetConfig, use_labels: bool) -> DataBundle:
    mean = (0.5,) * cfg.in_channels
    std = (0.5,) * cfg.in_channels
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    fid_transform = transforms.Compose([transforms.ToTensor()])
    train_ds = ImageFolderDataset(
        path=cfg.data_dir,
        use_labels=use_labels,
        cache=False,
        transform=train_transform,
    )
    fid_ds = ImageFolderDataset(
        path=cfg.data_dir,
        use_labels=use_labels,
        cache=False,
        transform=fid_transform,
    )
    train_loader = ResumableDataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        pin_memory=True,
    )
    fid_loader = DataLoader(
        fid_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    num_classes = cfg.label_dim if use_labels else 0
    return DataBundle(train=train_loader, fid=fid_loader, num_classes=num_classes)


def build_dataloaders(cfg: DatasetConfig) -> DataBundle:
    name = cfg.name.lower()
    if name == "cifar10":
        return _build_cifar10(cfg)
    if name == "mnist":
        return _build_mnist(cfg, datasets.MNIST)
    if name in {"fashion_mnist", "fmnist"}:
        return _build_mnist(cfg, datasets.FashionMNIST)
    if name == "imagenet":
        return _build_imagefolder(cfg, use_labels=True)
    if name == "ffhq":
        return _build_imagefolder(cfg, use_labels=False)
    raise NotImplementedError(f"Unsupported dataset {cfg.name}")
