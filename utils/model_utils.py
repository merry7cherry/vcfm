import copy
import pickle

import dnnlib
import torch
from omegaconf import DictConfig

from models.vcfm import GaussianCoupling, VariationallyCoupledFlowMatching
from networks.edm_networks import SongUNet
from networks.networks_edm2 import EDM2
from torch_utils import misc


def get_neural_net(
    cfg: DictConfig,
    *,
    out_channels: int,
    reload_url: str,
) -> torch.nn.Module:
    if cfg.model.class_conditional:
        label_dim = cfg.dataset.label_dim
    else:
        label_dim = 0

    if cfg.network.name == 'edm':
        net = SongUNet(
            img_resolution=cfg.dataset.img_resolution,
            in_channels=cfg.dataset.in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            embedding_type=cfg.network.embedding_type,
            encoder_type=cfg.network.encoder_type,
            decoder_type=cfg.network.decoder_type,
            channel_mult_noise=cfg.network.channel_mult_noise,
            resample_filter=list(cfg.network.resample_filter),
            model_channels=cfg.network.model_channels,
            channel_mult=list(cfg.network.channel_mult),
            dropout=cfg.network.dropout,
            num_blocks=cfg.network.num_blocks,
        )
    elif cfg.network.name == 'edm2':
        net = EDM2(
            img_resolution=cfg.dataset.img_resolution,
            in_channels=cfg.dataset.in_channels,
            out_channels=out_channels,
            label_dim=label_dim,
            model_channels=cfg.network.model_channels,
            channel_mult=cfg.network.channel_mult,
            dropout=cfg.network.dropout,
            dropout_res=cfg.network.dropout_res,
            num_blocks=cfg.network.num_blocks,
        )
    else:
        raise NotImplementedError(f"Unsupported network {cfg.network.name}")

    if reload_url:
        with dnnlib.util.open_url(reload_url) as f:
            data = pickle.load(f)
            if cfg.network.name == 'edm':
                misc.copy_params_and_buffers(
                    src_module=data['ema'].model,
                    dst_module=net,
                    require_all=False,
                )
            elif cfg.network.name == 'edm2':
                misc.copy_params_and_buffers(
                    src_module=data['ema'],
                    dst_module=net,
                    require_all=True,
                )
        if cfg.network.name == 'edm2':
            net = net  # EDM2 weights already loaded
    return net


def get_model(cfg: DictConfig):
    assert cfg.model.name == 'vcfm'
    assert cfg.network.name in ['edm', 'edm2']

    velocity_net = get_neural_net(
        cfg,
        out_channels=cfg.dataset.out_channels,
        reload_url=cfg.network.reload_url,
    )

    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.network.reload_url = ''
    cfg_copy.dataset.out_channels = cfg.dataset.out_channels * 2
    coupling_net = get_neural_net(
        cfg_copy,
        out_channels=cfg_copy.dataset.out_channels,
        reload_url='',
    )

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