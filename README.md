# VC-FM: Variationally-Coupled Flow Matching

This repository contains a minimal PyTorch implementation of **Variationally-Coupled Flow Matching (VC-FM)**. The method trains a flow-matching velocity field together with a variational coupling network that learns optimal transport pairings. EDM and EDM2 U-Nets are available as interchangeable backbones.

## What's inside

* Plain PyTorch training loop with two optimizers (velocity and coupling) and optional EMA.
* Linear-interpolant flow matching loss combined with the straightness regularizer for the coupling network.
* Dataset factory covering CIFAR-10, MNIST, Fashion-MNIST, FFHQ (image folder), and ImageNet (image folder) with conditional labels when requested.
* Lightweight callback system for periodic sample generation and FID evaluation.
* Inference helpers for loading checkpoints and generating samples without Lightning or WandB.

## Installation

Create an environment with the minimal dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

The repository bundles the original EDM/EDM2 `dnnlib` and `torch_utils` helper modules; no extra installation is required. CUDA-enabled PyTorch builds are recommended for practical training.

## Configuration

Configurations are simple OmegaConf/YAML files. The top-level [`conf/config.yaml`](conf/config.yaml) selects the dataset, model, and network definitions:

```yaml
defaults:
  - dataset: cifar10
  - model: vcfm
  - network: edm

training:
  log_every: 100
  eval_every: 1000
  sample_every: 1000
  sample_steps: 50
  grad_clip: null
  output_dir: runs
  seed: 42

callbacks:
  generate: true
  fid: true
```

Dataset- and network-specific options live in `conf/dataset/*.yaml` and `conf/network/*.yaml`. For example, switching to EDM2 simply replaces `network: edm` with `network: edm2`. Set `model.class_conditional: true` to enable label conditioning; the trainer automatically handles one-hot conversion.

## Training

Run training with the new CLI:

```bash
python main.py --config conf/config.yaml --output runs/cifar10 --device auto --save checkpoints/cifar10.pth
```

Key flags:

* `--config`: path to a YAML config (defaults to `conf/config.yaml`).
* `--device`: `auto`, `cpu`, or a CUDA device specifier such as `cuda:0`.
* `--output`: directory where samples and metrics are stored.
* `--save`: optional checkpoint path (`.pth`) saved after training.

Samples are written under `<output>/samples/` using the requested `sample_shape`. If `callbacks.fid=true` and the dataset provides a validation loader, the trainer computes FID every `training.eval_every` steps using the trained model (or EMA if enabled).

### Examples

* **EDM backbone on CIFAR-10**: keep the default config in `conf/config.yaml`.
* **EDM2 backbone**: set `network: edm2` in `conf/config.yaml` or pass `network=edm2` inside the YAML.
* **Conditional training**: set `model.class_conditional: true` and ensure the dataset exposes labels (e.g., CIFAR-10, MNIST).
* **FFHQ/ImageNet**: point `dataset.data_dir` to an image folder prepared following the EDM tooling. The loader supports optional labels through `dataset.label_dim`.

## Inference

Use `vcfm.inference.load_checkpoint` to reload a trained model and (optionally) its EMA copy:

```python
import torch
from vcfm.inference import load_checkpoint, generate

cfg, model, ema_model = load_checkpoint("checkpoints/cifar10.pth", device=torch.device("cuda"))
model.eval()
with torch.no_grad():
    samples = generate(ema_model or model, sample_shape=(16, 3, 32, 32), n_iters=50)
```

## Project structure

```
vcfm/
  config.py          # dataclass configuration loader
  data.py            # dataset/dataloader factory
  trainer.py         # training loop with two optimizers and EMA
  callbacks.py       # minimal callback system (sampling, FID)
  model_factory.py   # EDM/EDM2 builders for velocity & coupling nets
  utils.py           # seed, channel helpers, resumable dataloader
  inference.py       # checkpoint loading and sampling helpers
models/
  vcfm.py            # VC-FM loss definitions and sampling routine
networks/            # EDM and EDM2 U-Net implementations
```

## References

* [Rectified Flow](https://arxiv.org/abs/2307.06264)
* [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)
* [EDM2](https://arxiv.org/abs/2311.18828)

This implementation is intentionally minimal: no Lightning, WandB, or Hydra runtime. Logging relies on tqdm progress bars and the callback outputs.
