# ColoriSR

This repository is a dual colorization and super resolution. An AWS `g5.xlarge` instance, which has 24GB of VRAM. Docker is required to reproduce the results of the experiments.

## Setup

### SR-to-Color

To reproduce the results of the `SR-to-Color` component of this project, follow these steps.

`cd` into the root directory of this repository.

Build the `SR-to-Color` image:

```shell
$ docker build -t sr-to-color -f sr_to_color/Dockerfile .
```

Create the container from the image:
```shell
$ docker run -dit --gpus all --name sr-to-color sr-to-color
```

Exec into the container:

```shell
$ docker exec -ti sr-to-color bash
```

In the container, you'll begin in `/workspace/Real-ESRGAN/`. You first need to download the MS-Coco data and preprocess it. To reproduce the results, no flags are needed; the default values will do. It's this script that differentiates the SR-to-Color module from Real-ESRGAN. PyTorch transformations generate pairs of Coco images, one that is grayscale low resolution and another that is in-color high resolution.

```shell
$ python preprocess_data.py \
  -io <PREPROCESSED DATA ROOT DIR> \
  -rs <RANDOM SEED>
  -sz <RESIZED IM SIZE>
```

After preprocessing has completed, run the generate metadata script.
```shell
python scripts/generate_meta_info_pairdata.py \
  --input datasets/coco/tr/hq datasets/coco/tr/lq \
  --meta_info datasets/coco/meta_info/colorizer_sr.txt
```

Finally, run the fine-tuning script. The `-opt` is customized to perform the task of dual colorization + super resolution.
```shell
python realesrgan/train.py -opt options/finetune_colorizer_sr.yml --auto_resume
```

### Color-to-SR

This module represents future work for this project. In this module, we aim to achieve dual colorization from pretraining a colorizing a U-Net based GAN before fine-tuning the generator and discriminator on low resolution, grayscale images. Thus far, we've pretrained the colorizer model from scratch. The next step is to do the fine tuning and compare the results to those SR-to-Color produced.

To run the pretraining, first `cd` into the root directory of this repository.

Build the `Color-to-SR` image:


```shell
$ docker build -t color_to_src -f color_to_src/Dockerfile .
```

Create the container from the image:
```shell
$ docker run -dit --gpus all --name color-to-sr color-to-sr
```

Exec into the container:

```shell
$ docker exec -ti color-to-sr bash
```


In the container, you'll begin in `/workspace/`. To initiate the pretraining, execute the following command. This step took over 10 hours to complete on an AWS `g5.xlarge` instance.

```shell
$ python -m color_to_sr.pretrain
```

Next, execute the `preprocess_data.py` script as instructed above.

At that point, you're all caught up! The next step for us is to write the fine-tuning script to complete the `color_to_sr` module.
