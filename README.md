# Towards Real-time Text-driven Image Manipulation with Unconditional Diffusion Models

Official implementation [](https://arxiv.org/abs/2011.13786) by ...

This code based on [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

![An image](./utils_imgs/readme.jpg)

## Overview

This work addresses efficiency of the recent text-driven editing methods based on unconditional diffusion
models and develop a novel algorithm that learns image manipulations 4.5−10× faster and applies them 8× faster.

![An image](./utils_imgs/overview-1.jpg)

We provide **two settings** for editing of images

**1.** _Domain adaptation setting_

It firstly fine-tunes diffusion model using set of images (about 50 examples) to learn transformation. 
Then diffusion model applies learned transformation to any image. The entire procedure takes about **45 secs** and **7 GiB** (tested on NVIDIA A100)

**2.** _Single image editing_

Diffusion model fine-tunes using only one image provided by user. And then transforms the image. For new images a new fine-tuning is needed. This setting takes about **4 secs**. 

This work uses diffusion models pretrained on Celeba-HQ, LSUN-Church, AFHQ-Dog and ImageNET datasets.

## Getting started

### 0. Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

First of all, try to play with our **colab notebook**. It is single image setting in which you can edit your own images.
### 1. Preparation

_Install required dependencies_
```
# Clone the repo
!git clone https://github.com/quickjkee/EffDiff

# Install dependencies
!pip install ftfy regex tqdm
!pip install lmdb
!pip install pynvml
!pip install git+https://github.com/openai/CLIP.git

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
```

_Download pretrained diffusion models_

* Pretrained diffuson models on CelebA-HQ, LSUN-Church are automatically downloaded in the code.

* For AFHQ-Dog and ImageNet please download corresponding models: [ImageNET](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt), [AFHQ-Dog](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&cid=72419B431C262344&id=72419B431C262344%21103832&parId=72419B431C262344%21103807&o=OneUp).
After downloading put them in ```/pretrained``` folder


_Download datasets (only for domain adaptation setting)_