from curses import raw
from pathlib import Path
from turtle import pos
import rawpy
from PIL import Image
from torch.utils.data import Dataset
import os
from io import BytesIO
import numpy as np
import torch 
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
import yaml
import itertools
import torch.nn.functional as F


def auto_padding(img, times=32):
    # img: torch tensor image with shape H*W*C
    b, c, h, w = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = F.pad(img, [w1, w2, h1, h2], "reflect")
    return img, [w1, w2, h1, h2]

def to_bchw_tensor(np_img):
    img_tensor = torch.tensor(np_img)
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(2)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    return img_tensor


def to_np_from_bchw_tensor(img_tensor):
    return img_tensor.squeeze(0).numpy()


def squeeze2d(input, factor=2):
    if isinstance(input, np.ndarray):
        input = to_bchw_tensor(input)
    assert factor >= 1 and isinstance(factor, int)
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)
    return x


def unsqueeze2d(input, factor=2):
    assert factor >= 1 and isinstance(factor, int)
    factor2 = factor ** 2
    if factor == 1:
        return input
    size = input.size()
    B = size[0]
    C = size[1]
    H = size[2]
    W = size[3]
    assert C % (factor2) == 0, "{}".format(C)
    x = input.view(B, C // factor2, factor, factor, H, W)
    x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
    x = x.view(B, C // (factor2), H * factor, W * factor)
    return x


def random_flip(img_list):
    random_choice = np.random.choice([True, False])
    res_list = []
    for img in img_list:
        img = img if random_choice else np.flip(img, 1).copy()
        res_list.append(img)
    return res_list


def random_crop_from_folder(low_folder_path, crop_size):
    with open(os.path.join(low_folder_path, "meta.yaml"), "r") as f:
        meta = yaml.load(f.read(), yaml.FullLoader)
        patch_size = meta["patch_size"]

    patch_list = []
    _, _, size_lr_x, size_lr_y = meta["shape_0"]
    start_x_lr = np.random.randint(low=0, high=(
        size_lr_x - crop_size) + 1) if size_lr_x > crop_size else 0
    start_y_lr = np.random.randint(low=0, high=(
        size_lr_y - crop_size) + 1) if size_lr_y > crop_size else 0

    for prefix in ["0", "1"]:
        b, c, size_lr_x, size_lr_y = meta["shape_%s"%prefix]
        x_start_file = (start_x_lr // patch_size) * patch_size  # x start index in file
        y_start_file = (start_y_lr // patch_size) * patch_size
        # b, c, size_lr_x, size_lr_y = meta["shape_%s" % prefix]
        patch_res = np.zeros([b, c, crop_size, crop_size], np.float32)
        i = 0
        while x_start_file + i * patch_size < start_x_lr + crop_size:
            j = 0
            while y_start_file + j * patch_size < start_y_lr + crop_size:
                raw_patch = np.load(os.path.join(low_folder_path, "%s_%d_%d.npy" % (
                    prefix, x_start_file + i * patch_size, y_start_file + j * patch_size)))

                def idx_func(start, start_file, idx): return [max(start - (start_file + idx * patch_size), 0), min(
                    (start + crop_size) - (start_file + idx * patch_size), patch_size)]
                i_start, i_end = idx_func(start_x_lr, x_start_file, i)
                j_start, j_end = idx_func(start_y_lr, y_start_file, j)
                mini_patch = raw_patch[:, :, i_start: i_end, j_start: j_end]
                def idx_res_func(start, start_file, idx): return [max(
                    idx * patch_size - (start - start_file), 0), min((idx + 1) * patch_size - (start - start_file), crop_size)]
                i_start_res, i_end_res = idx_res_func(start_x_lr, x_start_file, i)
                j_start_res, j_end_res = idx_res_func(start_y_lr, y_start_file, j)
                patch_res[:, :, i_start_res: i_end_res,
                          j_start_res: j_end_res] = mini_patch.astype(np.float32)
                j += 1
            i += 1
        patch_list.append(torch.tensor(patch_res))
    return patch_list


def random_crop(img_list, size):
    shape = img_list[0].shape
    assert len(shape) == 4  # B*C*H*W
    for img in img_list:
        assert isinstance(img_list[0], torch.Tensor) and img.shape[2:] == shape[2:]

    if size is not None:
        size_lr_x = shape[2]
        size_lr_y = shape[3]

        patch_list = []
        start_x_lr = np.random.randint(low=0, high=(
            size_lr_x - size) + 1) if size_lr_x > size else 0
        start_y_lr = np.random.randint(low=0, high=(
            size_lr_y - size) + 1) if size_lr_y > size else 0
        for img in img_list:
            patch = img[:, :, start_x_lr:start_x_lr +
                        size, start_y_lr:start_y_lr + size]
            patch_list.append(patch)
        return patch_list
    return img_list


class RawImageDataset(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        self.file_path = os.path.join(root, split + ".txt")
        self.split = split
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        self.samples = []
        for line in lines:
            line = line.rstrip()
            self.samples.append(line.split(", "))  # "rgb_path, raw_path" or "raw_path"
        if split != "train":
            self.samples = self.samples[:16]
        self.transform = transform

    def get_format_and_fname(self, path):
        basename = os.path.basename(path)
        fname, format = basename.split(".")
        return fname, format

    def get_patch_folder_path(self, fpath):
        return os.path.join("datasets", "np_raw_patches", "_".join(fpath.removeprefix("./").split("/")))

    def save_patch_to_folder(self, tensor_list, folder_path, patch_size=512):
        """
        tensor_list: the list of 1*C*H*W torch.Tensor
        folder_path: 
        """
        assert tensor_list[0].shape[2:] == tensor_list[1].shape[2:]
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "meta.yaml"), "w") as f:
            yaml.dump({"shape_0": list(tensor_list[0].shape), "shape_1": list(
                tensor_list[1].shape), "patch_size": patch_size}, f)
        for idx, tensor in enumerate(tensor_list):
            b, c, h, w = tensor.shape
            h_list = list(np.arange(0, h, patch_size))
            w_list = list(np.arange(0, w, patch_size))
            for h_start, w_start in itertools.product(h_list, w_list):
                np.save(os.path.join(folder_path, "%d_%d_%d.npy" % (idx, h_start, w_start)),
                        tensor[:, :, h_start:h_start + patch_size, w_start:w_start + patch_size].numpy())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        raw_img = None
        # get raw_path and jpeg_rgb_path
        if len(self.samples[index]) == 1:
            raw_path = self.samples[index][0]
        else:
            rgb_path, raw_path = self.samples[index]
            
            
        patch_folder_path = self.get_patch_folder_path(raw_path)
        if not os.path.exists(patch_folder_path):
            if len(self.samples[index]) == 1:
                raw_path = self.samples[index][0]
                # fname_raw, format_raw = self.get_format_and_fname(raw_path)
                # postprocess RAW image
                raw_img = rawpy.imread(raw_path)
                rgb_jpeg = raw_img.postprocess(
                    use_camera_wb=True, half_size=False, no_auto_bright=True)
                format_rgb = "lossless"
                image = Image.fromarray(rgb_jpeg)
            else:
                rgb_path, raw_path = self.samples[index]
                fname_rgb, format_rgb = self.get_format_and_fname(rgb_path)
                # fname_raw, format_raw = self.get_format_and_fname(raw_path)
                image = Image.open(fname_rgb)

            # if the RGB image is lossless, then compress it using JPEG
            if format_rgb in ["tif", "lossless"]:
                output = BytesIO()
                image.save(output, 'JPEG', quality=90)
                output.seek(0)
                rgb_jpeg = Image.open(output)
            elif format_rgb in ["jpg", "JPG", "JPEG"]:
                rgb_jpeg = Image.open(fname_rgb).convert("RGB")
            else:
                raise NotImplementedError(
                    "The format %s is not implmented." % (fname_rgb))
            
            rgb_jpeg = rgb_jpeg.convert("RGB")
            rgb_jpeg = to_tensor(rgb_jpeg).unsqueeze(0)

            # read RAW image
            if raw_img is None:
                raw_img = rawpy.imread(raw_path)

            raw_visible = raw_img.raw_image_visible.astype(np.float32)
            # if raw_visible.s
            # lr_color_index = lr_raw.raw_colors_visible.astype(np.float32)
            assert np.std(raw_img.black_level_per_channel) < 2
            black_level = raw_img.black_level_per_channel[0]
            raw_visible = (raw_visible - black_level) / \
                (raw_img.white_level - black_level)
            if raw_visible.shape == rgb_jpeg.shape[4:1:-1]:
                raw_visible = raw_visible.transpose(1, 0)
            # lr_tensor = torch.tensor(lr_visible)
            # raw_visible = squeeze2d(raw_visible)
            raw_visible = squeeze2d(raw_visible) # To make sure that have the fixed cmos pattern, e.g.g, RGGB
            img_list = [rgb_jpeg, raw_visible]
            if self.split == "train":
                img_list[0] = squeeze2d(img_list[0])
                print("save patches to file")
                self.save_patch_to_folder(img_list, patch_folder_path)
            raw_img.close()
            output.close()
        if self.split == "train":
            # H, W = raw_visible.shape[0], raw_visible.shape[1]
            img_list = random_crop_from_folder(patch_folder_path, crop_size=1024)
            img_list[0] = unsqueeze2d(img_list[0])
        else:
            img_list[0] = auto_padding(img_list[0],128)[0]
            img_list[1] = auto_padding(img_list[1],64)[0]
        # Squeeze
        for idx, img in enumerate(img_list):
            img_list[idx] = img.squeeze(0)
        return img_list

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    dataset = RawImageDataset("./datasets/SID", split="val")
    # dataset = RawImageDataset("./datasets/fivek_dataset", split="train")
    img_list = dataset[0]
    pass
