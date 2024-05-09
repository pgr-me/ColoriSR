import argparse
import os
import glob
from pathlib import Path
import time

from fastai.data.external import untar_data, URLs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import torch
from torch import nn, optim
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange

from color_to_sr.models import *

# Defaults
SIZE = 256
DST_DIR = "experiments/results/color_to_sr"
RANDOM_SEED = 123
LR_G = 2e-4
LR_D = 2e-4
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100.

def argparser():
    parser = argparse.ArgumentParser(prog="Pretrain colorizer model from scratch.")
    io_help = "Directory where pretraining outputs, including models and visualizations, are saved."
    rs_help = "Random seed to fix sampling."
    sz_help = "Output image size."
    parser.add_argument("-io", "--dst_dir", type=Path, default=DST_DIR, help=io_help)
    parser.add_argument("-rs", "--random_seed", default=RANDOM_SEED, type=int, help=rs_help)
    parser.add_argument("-sz", "--size", default=SIZE, type=int, help=sz_help)
    return parser.parse_args()



if __name__ == "__main__":
    args = argparser()
    args.dst_dir.mkdir(exist_ok=True, parents=True)

    np.random.seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Download MS Coco data.")
    coco_path = untar_data(URLs.COCO_SAMPLE)
    coco_path = coco_path / "train_sample" 

    print("Define pathing and tr, te, val splits.")
    paths = list(coco_path.glob("*.jpg"))
    np.random.seed(123)
    paths_subset = np.random.choice(paths, 10_000, replace=False) # choosing 1000 images randomly
    rand_idxs = np.random.permutation(10_000)
    train_idxs = rand_idxs[:8000] # choosing the first 8000 as training set
    val_idxs = rand_idxs[8000:9000] # choosing last 2000 as validation set
    te_idxs = rand_idxs[10_000:] # choosing last 2000 as validation set
    train_paths = paths_subset[train_idxs]
    val_paths = paths_subset[val_idxs]
    te_paths = paths_subset[te_idxs]

    print("Define datasets and dataloaders.")
    tr_dataset = ColorizationDataset(paths=train_paths, split="train")
    val_dataset = ColorizationDataset(paths=val_paths, split="val")

    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=False)

    print("Initialize model for pretraining.")
    best_fid = 1e9
    model = MainModel(
        net_G=None,
        lr_G=LR_G,
        lr_D=LR_D,
        beta1=BETA1,
        beta2=BETA2,
        lambda_L1=LAMBDA_L1
    )
    n_val_batches = len(val_loader)
    tr_loss_results = []
    val_fids = []
    for e in range(EPOCHS):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"\nEpoch {e+1}/{EPOCHS}")
        
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"[{e+1}]: Training loop")
        loss_meter_dict = create_loss_meters() # function returing a dictionary of objects to 
        i = 0                                  # log the losses of the complete network
        for tr_data in tqdm(tr_loader, desc="Training"):
            model.setup_input(tr_data) 
            model.optimize()
            update_losses(model, loss_meter_dict, count=tr_data['L'].size(0)) # function updating the log objects
            i += 1

        log_results(loss_meter_dict) # function to print out the losses
        tr_loss_results.append({k: v.avg for k, v in loss_meter_dict.items()})

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print(f"[{e+1}]: Validation loop")
        model.eval()
        tot_fid = 0
        for val_data in tqdm(val_loader, desc="Validation"):
            val_real_L = val_data['L'].to(device)
            val_real_ab = val_data['ab'].to(device)
            with torch.no_grad():
                val_fake_ab = model.net_G(val_real_L)
            val_real = lab_to_rgb(val_real_L, val_real_ab)
            val_fake = lab_to_rgb(val_real_L, val_fake_ab)
            val_real = torch.moveaxis(torch.from_numpy(np.clip(val_real * 255, a_min=0, a_max=255).astype(np.uint8)), 3, 1)
            val_fake = torch.moveaxis(torch.from_numpy(np.clip(val_fake * 255, a_min=0, a_max=255).astype(np.uint8)), 3, 1)
            # Update with real and generated images
            val_fid = FrechetInceptionDistance(feature=64)
            val_fid.update(val_real, real=True)
            val_fid.update(val_fake, real=False)
            tot_fid += val_fid.compute().detach().cpu().item()

        mean_fid = tot_fid / (n_val_batches * BATCH_SIZE)
        val_fids.append(dict(epoch=e+1, fid=mean_fid))
        if mean_fid < best_fid:
            best_fid = mean_fid
            best_model = model.state_dict()
            print(f"[{e+1}]: Best FID: {best_fid:.5f}")
            torch.save(best_model, args.dst_dir / f"{e+1}_best_model.pth")
            visualize(model, val_data, epoch=e+1, save=True)
        model.train()

    # train_model(model, tr_loader, val_loader, 100)
    tr_loss_results_df = pd.DataFrame(tr_loss_results)
    tr_loss_results_df.to_csv(args.dst_dir / "tr_loss_results.csv", index=False)

    val_fids_df = pd.DataFrame(val_fids)
    val_fids_df.to_csv(args.dst_dir / "val_fids.csv", index=False)
