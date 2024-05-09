import argparse

import os
import glob
from pathlib import Path
import time
from typing import List

from fastai.data.external import untar_data, URLs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
from skimage import transform as skimage_transform
import torch
from torch import nn, optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F, Compose
from tqdm import tqdm, trange

# Defaults
SIZE = 256
ROOT_DIR = "datasets/coco"
RANDOM_SEED = 123


def argparser():
    parser = argparse.ArgumentParser(prog="SR colorizer data preprocessor.")
    io_help = "Path to root data directory where tr, te, and val data saved."
    rs_help = "Random seed to fix sampling."
    sz_help = "Output image size."
    parser.add_argument("-io", "--root_dir", type=Path, default=ROOT_DIR, help=io_help)
    parser.add_argument("-rs", "--random_seed", default=RANDOM_SEED, type=int, help=rs_help)
    parser.add_argument("-sz", "--size", default=SIZE, type=int, help=sz_help)
    return parser.parse_args()


class SynthesizedDataset(Dataset):
    def __init__(self, srcs: List[Path], lowres_transform, highres_transform):
        self.srcs = srcs
        self.lowres_transform = lowres_transform
        self.highres_transform = highres_transform

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, idx):
        img_path = self.srcs[idx]
        image = Image.open(img_path)
        lowres_image = self.lowres_transform(image)
        highres_image = self.highres_transform(image)
        return lowres_image, highres_image


def make_dataset_dirs(root_dir: Path):
    for mode in ("tr", "val", "te"):
        mode_dir = root_dir / mode
        lowres_dir = mode_dir / "lq"
        highres_dir = mode_dir / "hq"
        lowres_dir.mkdir(exist_ok=True, parents=True)
        highres_dir.mkdir(exist_ok=True, parents=True)


def preprocess_data(root_dir: Path, dataset: Dataset, mode: str):
    assert mode in ("tr", "te", "val"), f"Mode {mode} must be either 'tr', 'te', or 'val'."
    mode_dir = root_dir / mode
    for ix, (lowres_img, highres_img) in enumerate(tqdm(dataset, desc=f"{mode} progress")):
        fn = dataset.srcs[ix].name
        lowres_dst = mode_dir / "lq" / fn
        highres_dst = mode_dir / "hq" / fn
        lowres_img.save(lowres_dst)
        highres_img.save(highres_dst)


def tensor_to_pil(tensor):
    return Image.fromarray(np.moveaxis((tensor.numpy()*255).astype(np.uint8), 0, -1))


if __name__ == "__main__":
    args = argparser()

    np.random.seed(args.random_seed)

    # Make directories
    make_dataset_dirs(args.root_dir)

    # Download Coco data using fastai
    coco_path = untar_data(URLs.COCO_SAMPLE)
    coco_path = coco_path / "train_sample"
    paths = list(coco_path.glob("*.jpg"))

    # Sample larger Coco dataset
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:9000] # choosing last 2000 as validation set
    te_idxs = rand_idxs[9000:10_000]
    train_paths = tr_srcs = paths_subset[train_idxs]
    val_paths = val_srcs = paths_subset[val_idxs]
    te_paths = te_srcs = paths_subset[te_idxs]

    # Define transforms to generate high res and grayscale low res image pairs
    lowres_transform = transforms.Compose([
        transforms.CenterCrop(args.size),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(args.size // 4),
    ])
    highres_transform = transforms.Compose([
        transforms.CenterCrop(args.size),
    ])

    # Define tr, val, and te datasets
    tr_dataset = SynthesizedDataset(tr_srcs, lowres_transform, highres_transform)
    val_dataset = SynthesizedDataset(val_srcs, lowres_transform, highres_transform)
    te_dataset = SynthesizedDataset(te_srcs, lowres_transform, highres_transform)

    # Preprocess datasets for use in Real-ESRGAN finetuning
    preprocess_data(args.root_dir, tr_dataset, "tr")
    preprocess_data(args.root_dir, te_dataset, "te")
    preprocess_data(args.root_dir, val_dataset, "val")

