# Towards Real-time Text-driven Image Manipulation with Unconditional Diffusion Models.

Official implementation [](https://arxiv.org/abs/2011.13786) by ...

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

![An image](./utils_imgs/readme_faces.jpg)

## Overview

This work addresses efficiency of the recent text-
driven editing methods based on unconditional diffusion
models and develop a novel algorithm that learns image ma-
nipulations 4.5−10× faster and applies them 8× faster.

![An image](./utils_imgs/overview-1.jpg)

## Getting started

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
