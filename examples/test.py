import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.chdir("../")
import math
import io
import torch
from compressai.datasets import RawImageDataset
import cv2
import torch.nn as nn
import torch.nn.functional as F
import os
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
# torch.use_deterministic_algorithms(True)
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image
from PIL import Image
import seaborn
import tqdm
import matplotlib.pyplot as plt
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from skimage.metrics import structural_similarity as cal_ssim
from compressai.datasets.dataset_raw import DatasetRAWTest, DatasetRAW
from pytorch_msssim import ms_ssim
from compressai.zoo import raw_hyperprior, raw_context, learned_context
# from ipywidgets import interact, widgets
import argparse
import thop

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(os.getcwd())
parser = argparse.ArgumentParser(description="Example training script.")
parser.add_argument(
    "-i",
    default="0",
    type=int,
    help="The model index"
)
parser.add_argument(
    "--max",
    default=0.01,
    type=float,
    help="maximum value of error bar"
)
parser.add_argument(
    "--lambda",
    default="0.05",
    dest="lmbda",
    type=float,
    help="maximum value of error bar"
)
parser.add_argument(
    "--gamma",
    default="0",
    type=float
)
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
    "--drop_num",
    default=0,
    type=int,
)
parser.add_argument(
    "--val",
    default="val_Sony", # val_Sony_jpg
    type=str,
)
parser.add_argument(
    "--ntest",
    default=-1, # val_Sony_jpg
    type=int,
)
parser.add_argument(
    "-q",
    default=-1, # val_Sony_jpg
    type=int,
    help="JPEG quality"
)

parser.add_argument(
    "--cache",
    action="store_true",
    help="load the dataset to memory"
)

parser.add_argument(
    "--raw_space",
    type=str,
    choices=["raw_linear", "xyz_gamma", "cvpr2022", "raw_gamma", "sid"],
    default="raw_linear",
)

parser.add_argument(
    "--file_type",
    type=str,
    help="load the dataset to memory",
    default="tif"
)

args = parser.parse_args()
print(args)
from config_files import *

ckpt = eval(args.g)[args.i]
# ckpt = nus_soft_jpg_lists[args.i]
model_basename = os.path.basename(ckpt['path'])+args.info
quality_str = "_q%d" % args.q if not ckpt["nocompress"] else ""
img_folder = f"results/error_map/{args.g}/{model_basename}/{args.val}{quality_str}{f'_gamma_{args.gamma:.1f}' if args.gamma!=0 else ''}{f'_drop_{args.drop_num}' if args.drop_num!=0 else ''}"
csv_folder = f"results/csv/{args.g}/{model_basename}"
os.makedirs(csv_folder, exist_ok=True)
result_csv = open(f"{csv_folder}/{args.val}{quality_str}{f'_gamma_{args.gamma:.1f}' if args.gamma!=0 else ''}{f'_drop_{args.drop_num}' if args.drop_num!=0 else ''}.csv","w")
result_csv.write(f"idx,PSNR,SSIM,bpp\n")

path, model, quality, noquant, demosaic, reduce_c, no_compress, channel_mean, discrete_context, use_deconv, only_demosaic = ckpt.pop("path"), ckpt.pop("model"), ckpt.pop("quality", 2), ckpt.pop("noquant", False), ckpt.get("demosaic", False), ckpt.pop("reduce_c", 1), ckpt.pop("nocompress", False), ckpt.get("channel_mean", False),  ckpt.pop("discrete_context", False), ckpt.pop("use_deconv", False), ckpt.pop("only_demosaic", False)

if args.cache:
    test_dataset = DatasetRAWTest(os.path.join(args.dataset, "test"), ftype=args.file_type)
else:
    test_dataset = RawImageDataset(args.dataset, split=args.val, transform=None, nocompress=no_compress, ntest=args.ntest, qlist=[args.q], raw_space=args.raw_space,**ckpt)

net = model(quality=quality, pretrained=False, noquant=noquant, reduce_c= reduce_c, discrete_context=discrete_context, use_deconv=use_deconv, gamma=args.gamma, drop_num=args.drop_num, **ckpt).eval()
ckpt_from_file = torch.load(path)
print(f"Loading the model from {path}")
print(f"The model is trained by {ckpt_from_file['epoch']} epoch")
net.load_state_dict(ckpt_from_file["state_dict"])

net = net.to(device)
print("The computational cost for 512*512 image:")
if ckpt.get("lambda_list", None) is not None:
    print(thop.clever_format(thop.profile(net, (torch.randn(1,3,512,512).to(device), torch.randn(1,3,512,512).to(device), 0.05)))) 
else:
    print(thop.clever_format(thop.profile(net, (torch.randn(1,3,512,512).to(device), torch.randn(1,3,512,512).to(device))))) 

def auto_padding(img, times=32):
    # img: numpy image with shape H*W*C
    b, c, h, w = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = F.pad(img, [w1, w2, h1, h2], "reflect")
    return img, [w1, w2, h1, h2]
psnr_list = []
bpp_list = []
ssim_list = []
re_list = []

net.update()
def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_re(target, rec):
    re = (target-rec).abs() / ((rec + target)/2).clip(0.001)
    return re.mean()

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()

def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    print("shape:", size)
    # size = unsqueeze2d(out_net['x_hat']).size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
            for likelihoods in out_net['likelihoods'].values()).item()
 
def remove_boarder(np_img):
    np_img_gray = np_img.mean(axis=2)
    w_sum = np.sum(np_img_gray, axis=0)
    h_sum = np.sum(np_img_gray, axis=1)
    h_area = np.argwhere(h_sum < h_sum.max())
    w_area = np.argwhere(w_sum < w_sum.max())
    return np_img[h_area.min():h_area.max(), w_area.min():w_area.max()]

def iter_str_length(list_x):
    string_length = []
    for x in list_x:
        if isinstance(x, (str, bytes)):
            string_length.append(len(x))
        else:
            string_length.extend(iter_str_length(x))
    return string_length

os.makedirs(img_folder, exist_ok=True)
# for i in tqdm.tqdm(range(len(test_dataset))):'
for i in tqdm.tqdm(range(len(test_dataset))):
    # if i not in [8]: continue
    rgb_jpeg, raw_visible, metadata = test_dataset[i]
    # rgb_jpeg, raw_visible = rgb_jpeg[:,:2920,:4386], raw_visible[:,:2920,:4386]
    rgb_jpeg, _ = auto_padding(rgb_jpeg.to(device).unsqueeze(0), 128)
    raw_visible, pad_params = auto_padding(raw_visible.to(device).unsqueeze(0), 128)
    import torch.cuda.amp as amp
    net.eval()
    
    with torch.no_grad():
        torch.cuda.empty_cache()

        raw_visible_ = raw_visible
        if max(list(raw_visible.shape)) > 4000:
            Bytes = 0
            rgb_jpeg_patches = rgb_jpeg.chunk(2, dim=3)
            raw_visible_patches = raw_visible.chunk(2, dim=3)
            rec_raw_patch_list = []
            for rgb_jpeg, raw_visible in zip(rgb_jpeg_patches, raw_visible_patches):
                with amp.autocast():
                    compressed_result = net.compress(raw_visible, rgb_jpeg, lmbda=args.lmbda)
                Bytes += sum(iter_str_length(compressed_result["strings"]))
                with amp.autocast():
                    rec_from_strings = net.decompress(compressed_result["strings"], compressed_result["shape"], rgb_jpeg, z_mean = compressed_result["z_mean"] if channel_mean else None, compressed_result = compressed_result, lmbda=args.lmbda)
                rec_raw_patch_list.append(rec_from_strings["x_hat"].float())
            rec_raw_ = torch.cat(rec_raw_patch_list, dim=3)
        else:
            with amp.autocast():
                compressed_result = net.compress(raw_visible, rgb_jpeg, lmbda=args.lmbda)
                Bytes = sum(iter_str_length(compressed_result["strings"])) # , 
                torch.cuda.empty_cache()
                
            with amp.autocast(): # https://github.com/pytorch/pytorch/issues/45910
                rec_from_strings = net.decompress(compressed_result["strings"], compressed_result["shape"], rgb_jpeg, z_mean = compressed_result["z_mean"] if channel_mean else None, compressed_result = compressed_result, lmbda=args.lmbda)
            
            rec_raw_ = rec_from_strings["x_hat"].float()

        
        
        kb = Bytes/1024
        print("Real size of compressed string %.5e KB"%kb)
        real_bpp = (8*Bytes/(raw_visible_.shape[2]*raw_visible_.shape[3]))
        print("Real bpp: %.5e"%real_bpp)
            
        rec_raw_ = F.pad(rec_raw_, [-p for p in pad_params])
        raw_visible_ = F.pad(raw_visible_, [-p for p in pad_params])
        
        psnr = compute_psnr(raw_visible_, rec_raw_)
        re = compute_re(raw_visible_, rec_raw_)
        ssim = cal_ssim(raw_visible_.squeeze().permute(1,2,0).cpu().numpy(), rec_raw_.squeeze().permute(1,2,0).cpu().numpy(), multichannel=True, channel_axis=2, data_range=1)
        print(f'MAE {(raw_visible_-rec_raw_).abs().mean().item()}')
        print(f'PSNR from strings: {psnr:.2f}dB')
        print(f'SSIM from strings: {ssim:.4f}')
        print(f'relative error from strings: {re:.4f}')

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        re_list.append(re.item())
        bpp_list.append(real_bpp)
        
        ax = seaborn.heatmap((raw_visible_-rec_raw_).cpu().abs().max(dim=1)[0].squeeze(0), xticklabels=False, yticklabels=False, vmax=args.max, cbar=False, vmin=0, square=True, linewidth = 0)
        fig = ax.get_figure()
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        im = data.reshape((int(h), int(w), -1))
        cv2.imwrite(f"{img_folder}/raw_{i}_{args.max}.jpg", cv2.cvtColor(remove_boarder(im), cv2.COLOR_RGB2BGR))
        fig.clf() 

        save_image(rec_raw_** (1 / 2.2), f"{img_folder}/raw_{i}.jpg")
        save_image(rgb_jpeg, f"{img_folder}/rsrgb_{i}.jpg")
        if metadata["wb"] != -1:
            awb = torch.tensor(metadata["wb"][:3])/metadata["wb"][1]
            save_image((rec_raw_.pow(2.2)/awb.to(rec_raw_.device)[:,None, None]).pow(1/2.2), f"{img_folder}/raw_{i}_woAW.jpg")
            save_image((raw_visible_.pow(2.2)/awb.to(rec_raw_.device)[:,None, None]).pow(1/2.2), f"{img_folder}/raw_{i}_woAW_gt.jpg")
        else:
            save_image(raw_visible_** (1 / 2.2), f"{img_folder}/raw_{i}_gt.jpg")
        
        result_csv.write(f"{i},{psnr},{ssim},{real_bpp}\n")
        
        del compressed_result
        del rec_from_strings

    
print(f"Mean PSNR: {np.mean(psnr_list):.5e} Mean SSIM: {np.mean(ssim_list):.5e} Mean RE: {np.mean(re_list):.5e} Mean BPP: {np.mean(bpp_list):.5e}\n\n")
result_csv.write(f"\n\n")
result_csv.write(f"Mean,{np.mean(psnr_list):.5e},{np.mean(ssim_list):.5e},{np.mean(bpp_list):.5e}\n")
result_csv.close()


        

