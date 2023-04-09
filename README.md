# Towards Real-time Text-driven Image Manipulation with Unconditional Diffusion Models
[[`Paper`](https://arxiv.org/abs/2011.13786)]

This code is based on [DiffusionCLIP](https://github.com/gwang-kim/DiffusionCLIP).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

![An image](./utils_imgs/readme.jpg)

## Overview

This work addresses efficiency of the recent text-driven editing methods based on unconditional diffusion
models and provides the algorithm that learns image manipulations 4.5−10× faster and applies them 8× faster than DiffusionCLIP.

![An image](./utils_imgs/overview-1.jpg)

We provide the following **two settings** for image manipulation:

**1.** _Prelearned image manipulations_

The pretrained diffusion model is adapted to the given textual transform on 50 images. 
Then, you can apply the learned transform to your images.
The entire procedure takes about **~45 secs** on NVIDIA A100.

**2.** _Single-image editing_

The pretrained diffusion model is adapted to your text description and image on the fly.
This setting takes about **~4 secs** on NVIDIA A100.

This work uses unconditional diffusion models pretrained on the **CelebA-HQ-256, LSUN-Church-256, AFHQ-Dog-256** and **ImageNet-512** datasets.

## Getting started

### 0. Colab notebook for single-image editing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rtu01eOB2gwr_j0gSyzXgkbMUKL_mNIx?usp=sharing)

This notebook provides a tool for single-image editing using our approach. You are welcome to edit your images according to any textual transform.
Please, pay close attention to the hyperparameter values.

### 1. Preparation

* _Install required dependencies_
```
# Clone the repo
!git clone https://github.com/quickjkee/eff-diff-edit

# Install dependencies
!pip install ftfy regex tqdm
!pip install lmdb
!pip install pynvml
!pip install git+https://github.com/openai/CLIP.git

conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
```

* _Download pretrained diffusion models_

  * Pretrained diffusion models on CelebA-HQ-256, LSUN-Church-256 are automatically downloaded in the code.

  * For AFHQ-Dog-256 and ImageNet-512, please download the corresponding models ([ImageNet](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt), [AFHQ-Dog](https://onedrive.live.com/?authkey=%21AOIJGI8FUQXvFf8&cid=72419B431C262344&id=72419B431C262344%21103832&parId=72419B431C262344%21103807&o=OneUp)) and put them into the ```./pretrained``` folder


* _Download datasets_ (this part can be skipped if you have your own training set, please see the second section for details)
   * For CelebA-HQ and AFHQ-Dog you can use the following code:    
  ```
  # CelebA-HQ 256x256
  bash data_download.sh celeba_hq .
  
  # AFHQ-Dog 256x256
  bash data_download.sh afhq .
  ```
  * For [LSUN-Church](https://www.yf.io/p/lsun) and [ImageNet](https://image-net.org/index.php), you can download them from the original sources and put them into `./data/lsun` or `./data/imagenet`.

### 2. Running
1. Select the config for the particular dataset: ```celeba.yaml / afhq.yaml / church.yaml / imagenet.yaml```.
2. Select the desired manipulation from the list. The list of available textual transforms for each dataset is [here](/utils/text_dic.py).\
Note that you can also add your own transforms to this file.
3. Check out the descriptions for the available options [here](/docs/clip-finetune-help).

Below we provide the commands for different settings:

* _Prelearned image manipulations (**dataset training** and **dataset test**)_ \
This command adapts the pretrained model using images from the training set and applies the learned transform to the test images. 
The following command uses 50 CelebA-HQ images for training and evaluation:

  ```
  python main.py --clip_finetune      \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr makeup       \
               --fast_noising_train 1   \
               --fast_noising_test 1    \
               --own_test 0             \
               --own_training 0         \
               --single_image 0         \
               --align_face 0           \
               --n_train_img 50         \
               --n_precomp_img 50       \
               --n_test_img 50          \
               --n_iter 5               \
               --t_0 350                \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 6          \
               --lr_clip_finetune 6e-6  \
               --id_loss_w 0.0          \
               --clip_loss_w 3          \
               --l1_loss_w 1.0 
    ```

* _Prelearned image manipulations (**dataset training** and **own test**)_ \
This command adapts the pretrained model using images from the training set and applies the learned transform to your own images. 
Basically, one needs to change ```--own_test 0``` to ```--own_test all```. 
Before running, put your images into the ```./imgs_for_test``` folder. 
Moreover, you can evaluate the learned transform on a single image: change ```--own_test all``` to ```--own_test <your_image_name>```.
 ```
  python main.py --clip_finetune      \
             --config celeba.yml      \
             --exp ./runs/test        \
             --edit_attr makeup       \
             --fast_noising_train 1   \
             --fast_noising_test 1    \
             --own_test all           \
             --own_training 0         \
             --single_image 0         \
             --align_face 0           \
             --n_train_img 50         \
             --n_precomp_img 50       \
             --n_test_img 50          \
             --n_iter 5               \
             --t_0 350                \
             --n_inv_step 40          \
             --n_train_step 6         \
             --n_test_step 6          \
             --lr_clip_finetune 6e-6  \
             --id_loss_w 0.0          \
             --clip_loss_w 3          \
             --l1_loss_w 1.0 
  ```
  
* _Prelearned image manipulations  (**own training** and **own test**)_\
  If you want to adapt a diffusion model on your own dataset then simply put them into the ```./imgs_for_train``` folder
  and change ```--own_training 0``` to ```--own_training 1```. In this case, you do not need to download any datasets above.
  ```
  python main.py --clip_finetune      \
             --config celeba.yml      \
             --exp ./runs/test        \
             --edit_attr makeup       \
             --fast_noising_train 1   \
             --fast_noising_test 1    \
             --own_test all           \
             --own_training 1         \
             --single_image 0         \
             --align_face 0           \
             --n_train_img 50         \
             --n_precomp_img 50       \
             --n_test_img 50          \
             --n_iter 5               \
             --t_0 350                \
             --n_inv_step 40          \
             --n_train_step 6         \
             --n_test_step 6          \
             --lr_clip_finetune 6e-6  \
             --id_loss_w 0.0          \
             --clip_loss_w 3          \
             --l1_loss_w 1.0 
  ```

* _Single-image editing (**own image**)_\
  To transform your own image in single image editing, change ```--single_image 0``` to ```--single_image 1```. 
  Then, put the image into ```./imgs_for_test``` and set up ```--own_test <your_image_name>```. For instance, ```--own_test girl.png``` as in the following example:
  ```
  python main.py --clip_finetune        \
               --config celeba.yml      \
               --exp ./runs/test        \
               --edit_attr makeup       \
               --fast_noising_train 1   \
               --fast_noising_test 1    \
               --own_test girl.png      \
               --own_training 1         \
               --single_image 1         \
               --align_face 1           \
               --n_train_img 1          \
               --n_precomp_img 1        \
               --n_test_img 1           \
               --n_iter 5               \
               --t_0 350                \
               --n_inv_step 40          \
               --n_train_step 6         \
               --n_test_step 6          \
               --lr_clip_finetune 6e-6  \
               --id_loss_w 0.0          \
               --clip_loss_w 3          \
               --l1_loss_w 1.0 
  ```
