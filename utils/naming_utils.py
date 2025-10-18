
from omegaconf import DictConfig
def get_run_name(cfg: DictConfig):
    name = f'{cfg.dataset.name}_{cfg.model.name}'

    if cfg.model.class_conditional:
        name += f'_cond'

    if cfg.network.reload_url:
        name += '_pretrained'

    name += f'_{cfg.network.name}'
    name += f'_stw_{cfg.model.straightness_weight}_kl_{cfg.model.kl_weight}'

    name += f'_bs_{cfg.dataset.batch_size * cfg.batch_multiplier}_drop_{cfg.network.dropout}'

    if cfg.gradient_clip_val > 0:
        name += f'_gclip_{cfg.gradient_clip_val}'

    if cfg.extra_name:
        name += f'_{cfg.extra_name}'


    return name
