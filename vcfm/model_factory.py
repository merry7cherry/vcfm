from __future__ import annotations

import copy
import pickle

import dnnlib
import torch

from models.vcfm import GaussianCoupling, VariationallyCoupledFlowMatching
from networks.edm_networks import SongUNet
from networks.networks_edm2 import EDM2
from torch_utils import misc

from .config import Config


def _build_velocity_net(cfg: Config, out_channels: int) -> torch.nn.Module:
    dataset = cfg.dataset
    net_cfg = cfg.network
    label_dim = dataset.label_dim if cfg.model.class_conditional else 0

    if net_cfg.name == "edm":
        net = SongUNet(
            img_resolution=dataset.img_resolution,
            in_channels=dataset.in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            embedding_type=net_cfg.embedding_type,
            encoder_type=net_cfg.encoder_type,
            decoder_type=net_cfg.decoder_type,
            channel_mult_noise=net_cfg.channel_mult_noise,
            resample_filter=list(net_cfg.resample_filter or [1, 1]),
            model_channels=net_cfg.model_channels,
            channel_mult=list(net_cfg.channel_mult or []),
            dropout=net_cfg.dropout,
            num_blocks=net_cfg.num_blocks,
        )
    elif net_cfg.name == "edm2":
        net = EDM2(
            img_resolution=dataset.img_resolution,
            in_channels=dataset.in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            model_channels=net_cfg.model_channels,
            channel_mult=net_cfg.channel_mult,
            dropout=net_cfg.dropout,
            dropout_res=net_cfg.dropout_res,
            num_blocks=net_cfg.num_blocks,
        )
    else:
        raise NotImplementedError(f"Unsupported network {net_cfg.name}")

    if net_cfg.reload_url:
        with dnnlib.util.open_url(net_cfg.reload_url) as f:
            data = pickle.load(f)
            if net_cfg.name == "edm":
                misc.copy_params_and_buffers(
                    src_module=data["ema"].model,
                    dst_module=net,
                    require_all=False,
                )
            elif net_cfg.name == "edm2":
                misc.copy_params_and_buffers(
                    src_module=data["ema"],
                    dst_module=net,
                    require_all=True,
                )
    return net


def build_model(cfg: Config) -> VariationallyCoupledFlowMatching:
    assert cfg.model.name == "vcfm"
    velocity_net = _build_velocity_net(cfg, cfg.dataset.out_channels)

    coupling_cfg = copy.deepcopy(cfg)
    coupling_out_channels = cfg.dataset.out_channels * 2
    coupling_cfg.dataset.out_channels = coupling_out_channels
    coupling_cfg.network.reload_url = ""
    coupling_net = _build_velocity_net(coupling_cfg, coupling_out_channels)

    label_dim = cfg.dataset.label_dim if cfg.model.class_conditional else 0
    coupling = GaussianCoupling(
        coupling_net,
        label_dim=label_dim,
        min_log_std=cfg.model.min_log_std,
        max_log_std=cfg.model.max_log_std,
    )

    model = VariationallyCoupledFlowMatching(
        velocity_net=velocity_net,
        coupling_net=coupling,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        straightness_weight=cfg.model.straightness_weight,
        kl_weight=cfg.model.kl_weight,
        label_dim=label_dim,
    )
    return model
