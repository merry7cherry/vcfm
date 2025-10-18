# [ICML 2025] Implementation of ``VC-FM: Variationally-Coupled Flow Matching''
This repository now houses the implementation of **VC-FM: Variationally-Coupled Flow Matching**, our Flow Matching method that combines EDM/EDM2 backbones with a variational coupling network trained via the straight-trajectory objective inspired by Rectified Flow.
- arXiv: https://arxiv.org/abs/2502.18197

## Requirements
This code uses Weights & Biases, and assumes that you have your wandb key in a file named `wandb_config.py` in a variable named `key=your_wandb_key`.
This code uses the following libraries:
```angular2html
pytorch 
torchvision 
torchaudio 
pytorch-cuda
lightning
torchmetrics[image]
scipy 
scikit-learn 
matplotlib 
wandb
hydra-core
POT
pyspng
```
You can check `requirements.txt` for the exact packases used in our experiments.
## Configurable datasets and networks
All datasets are configured via Hydra in [`conf/dataset`](conf/dataset) and can be selected at the command line:

| Dataset key        | Notes                          |
|--------------------|--------------------------------|
| `cifar10`          | 32×32 RGB, 10 classes          |
| `mnist`            | 28×28 grayscale, 10 classes    |
| `fashion_mnist`    | 28×28 grayscale, 10 classes    |
| `ffhq`             | 64×64 RGB, optional labels     |
| `imagenet`         | 64×64 RGB, 1000 classes        |

Backbone networks are selected from [`conf/network`](conf/network) with `network=edm` (Song U-Net) or `network=edm2` (EDM2 U-Net). Both backbones are fully supported by the VC-FM objective and can optionally load pretrained weights through `network.reload_url`.

### Enabling class-conditional training
Set `model.class_conditional=True` to activate label conditioning. The Lightning module automatically converts integer labels to one-hot vectors with the correct dimensionality, and all datamodules expose label tensors whenever class conditioning is requested. During sampling you may provide either integer class indices or one-hot vectors; if omitted, zero vectors are used (equivalent to classifier-free guidance with a single null label).

## Training
We provide example commands for training VC-FM with the new Flow Matching objectives. Batch size is specified per device.

### CIFAR-10 with EDM backbone
```bash
python main.py \
    project=vcfm_cifar \
    dataset=cifar10 \
    dataset.num_workers=16 \
    dataset.batch_size=256 \
    model=vcfm \
    network=edm \
    model.straightness_weight=1.0 \
    model.kl_weight=1.0
```

### CIFAR-10 with EDM2 backbone and pretrained initialization
```bash
python main.py \
    project=vcfm_cifar_edm2 \
    dataset=cifar10 \
    dataset.num_workers=16 \
    dataset.batch_size=128 \
    model=vcfm \
    network=edm2 \
    network.reload_url='https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl' \
    model.straightness_weight=0.5 \
    model.kl_weight=0.5
```

### Class-conditional CIFAR-10
```bash
python main.py \
    project=vcfm_cifar_conditional \
    dataset=cifar10 \
    dataset.batch_size=256 \
    model=vcfm \
    model.class_conditional=True \
    network=edm \
    model.straightness_weight=1.0 \
    model.kl_weight=1.0
```

### FFHQ 64×64
Follow the dataset preparation instructions from [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm). Then run:
```bash
python main.py \
    project=vcfm_ffhq \
    dataset=ffhq \
    dataset.data_dir='your_data_dir' \
    dataset.batch_size=64 \
    model=vcfm \
    network=edm2 \
    model.class_conditional=False \
    model.straightness_weight=1.0 \
    model.kl_weight=0.5
```
## References
Parts of the code were adapted from the following codebases:
- [https://github.com/NVlabs/edm](https://github.com/NVlabs/edm)
- [https://github.com/locuslab/ect](https://github.com/locuslab/ect)
- [https://github.com/Kinyugo/consistency_models](https://github.com/Kinyugo/consistency_models)
- [https://github.com/atong01/conditional-flow-matching](https://github.com/atong01/conditional-flow-matching)
- [https://github.com/NVlabs/edm2](https://github.com/NVlabs/edm2)

## Contact
- Gianluigi Silvestri: gianlu.silvestri@gmail.com
- Chieh-Hsin (Jesse) Lai: Chieh-Hsin.Lai@sony.com
