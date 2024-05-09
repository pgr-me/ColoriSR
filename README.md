# ColoriSR

This repository is a dual colorization and super resolution.

## Setup

### SR-to-Color

To reproduce the results of the `SR-to-Color` component of this project, follow these steps.

`cd` into the root directory of this repository.

Build the `SR-to-Color` image:

```shell
$ docker build -t sr-to-color -f sr-to-color/Dockerfile .
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

TODO
