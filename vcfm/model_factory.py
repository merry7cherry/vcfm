from __future__ import annotations

import pickle

import dnnlib
import torch

from models.vcfm import LatentEncoder, VariationallyCoupledFlowMatching
from networks.edm_networks import SongUNet
from networks.networks_edm2 import EDM2
from torch_utils import misc

from .config import Config


def _build_velocity_net(cfg: Config, out_channels: int) -> torch.nn.Module:
    dataset = cfg.dataset
    net_cfg = cfg.network
    label_dim = dataset.label_dim if cfg.model.class_conditional else 0

    print(f"Building net with num_blocks: {net_cfg.num_blocks}")

    latent_dim = cfg.model.latent_dim

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
            latent_dim=latent_dim,
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
            latent_dim=latent_dim,
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


def _build_latent_encoder(cfg: Config) -> LatentEncoder:
    dataset = cfg.dataset
    label_dim = dataset.label_dim if cfg.model.class_conditional else 0
    return LatentEncoder(
        in_channels=dataset.in_channels,
        latent_dim=cfg.model.latent_dim,
        hidden_channels=cfg.model.phi_hidden_channels,
        num_layers=cfg.model.phi_num_layers,
        label_dim=label_dim,
    )


def build_model(cfg: Config) -> VariationallyCoupledFlowMatching:
    assert cfg.model.name == "vcfm"
    print(f"Building generation model with num_blocks: {cfg.network.num_blocks}")
    velocity_net = _build_velocity_net(cfg, cfg.dataset.out_channels)
    label_dim = cfg.dataset.label_dim if cfg.model.class_conditional else 0
    latent_encoder = _build_latent_encoder(cfg)

    model = VariationallyCoupledFlowMatching(
        velocity_net=velocity_net,
        latent_encoder=latent_encoder,
        sigma_min=cfg.model.sigma_min,
        sigma_max=cfg.model.sigma_max,
        flow_matching_theta_weight=cfg.model.flow_matching_theta_weight,
        straightness_weight=cfg.model.straightness_weight,
        kl_phi_weight=cfg.model.kl_phi_weight,
        label_dim=label_dim,
        latent_dim=cfg.model.latent_dim,
    )
    return model
