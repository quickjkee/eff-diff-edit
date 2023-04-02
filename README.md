# Towards Real-time Text-driven Image Manipulation with Unconditional Diffusion Models.

Official implementation [](https://arxiv.org/abs/2011.13786) by ...

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

![An image](./utils_imgs/readme_faces.jpg)

## Overview

This work addresses efficiency of the recent text-driven editing methods based on unconditional diffusion
models and develop a novel algorithm that learns image manipulations 4.5−10× faster and applies them 8× faster.

![An image](./utils_imgs/overview-1.jpg)

We provide **two settings** for editing of images

**1.** _Domain adaptation setting_

It firstly fine-tunes diffusion model using about 50 images to learn transformation. 
Then diffusion model applies learned transformation to any image. The entire procedure takes about **45 secs** (tested on A100)

**2.** _Single image editing_

Diffusion model fine-tunes using only single image provided by user. And then transforms the image. This setting takes about **4 secs**. But the first setting is more robust.

This work uses diffusion models pretrained on Celeba-HQ, LSUN-Church, AFHQ-Dog and ImageNET datasets.

## Getting started

### 0. Colab notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

First of all, try to play with our **colab notebook**. It is single image setting in which you can edit your own images.
### 1. Preparation

Install required dependencies
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

Pretrained diffusion models
