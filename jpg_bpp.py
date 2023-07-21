from email.policy import default
from genericpath import exists
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.chdir("../")
import math
import io
import torch
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import seaborn
import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as cal_ssim
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
from pytorch_msssim import ms_ssim
from compressai.zoo import raw_hyperprior,invcompress, raw_context, learned_context
from ipywidgets import interact, widgets
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(os.getcwd())
parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument(
    "-i",
    default="0",
    type=int,
    help="The model index"
),
parser.add_argument(
    "--max",
    default="0.01",
    type=float,
    help="maximum value of error bar"
),
parser.add_argument(
    "--info",
    default="",
    type=str,
)
parser.add_argument(
    "-g",
    default="nus_soft_jpg_lists",
    type=str,
    help="The experiments list name"
)
parser.add_argument(
    "--dataset",
    default="./datasets/NUS"
)
parser.add_argument(
    "--val",
    default="val_Olympus_f0", # val_Sony_jpg
    type=str,
)
parser.add_argument(
    "--ntest",
    default=-1, # val_Sony_jpg
    type=int,
)
parser.add_argument(
    "-q",
    default=10, # val_Sony_jpg
    type=int,
    help="JPEG quality"
)
args = parser.parse_args()

from compressai.datasets import RawImageDataset
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os

test_dataset = RawImageDataset(args.dataset, split=args.val, transform=None, nocompress=False, ntest=args.ntest, qlist=[args.q], only_demosaic=False)


bpp_list = []

# os.makedirs(img_folder, exist_ok=True)
for i in tqdm.tqdm(range(len(test_dataset))):
    # if i not in [49]: continue
    _, metadata = test_dataset[i]
    bpp_list.append(metadata["jpg_bpp"])
    print(metadata["jpg_bpp"])
print(np.mean(bpp_list))