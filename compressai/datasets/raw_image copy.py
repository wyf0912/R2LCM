from pathlib import Path
from turtle import pos
import rawpy
from PIL import Image
from torch.utils.data import Dataset
import os
from io import BytesIO
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, rotate
from torchvision.utils import save_image
import yaml
import itertools
import torch.nn.functional as F
import cv2
import torchvision.transforms as TF
import random
import exifread

def get_image_rotation(img_path):
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)
        # 获取Orientation标签
        orientation = tags['Image Orientation']
        if orientation.values[0] in ['Horizontal (normal)', 1]:
            return 0
        elif orientation.values[0] in ['Rotated 180', 3]:
            return 180
        elif orientation.values[0] == ['Rotated 270 CW', 7]:
            return 270
        elif orientation.values[0] == ['Rotated 90 CW', 5]:
            return 90
        else:
            # 如果没有Orientation标签，返回默认值0
            print(orientation.values[0], orientation)
            return 0

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
    assert shape[2] > size and shape[3] > size, shape
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

def center_crop(img_list, size):
    shape = img_list[0].shape
    assert len(shape) == 4  # B*C*H*W
    for img in img_list:
        assert isinstance(img_list[0], torch.Tensor) and img.shape[2:] == shape[2:]

    if size is not None:
        size_lr_x = shape[2]
        size_lr_y = shape[3]

        patch_list = []
        start_x_lr = (size_lr_x - size)//2
        start_y_lr = (size_lr_y - size)//2
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

    def __init__(self, root, transform=None, split="train", in_memory=False, ntest=16, ntrain=-1, repeat=1, **kwargs):
        self.file_path = os.path.join(root, split + ".txt")
        self.split = split
        self.demosaic = kwargs.pop("demosaic", False)
        self.xyz_space = kwargs.pop("xyz", False)
        self.only_demosaic = kwargs.pop("only_demosaic", False)
        self.nocompress = kwargs.pop("nocompress", False)
        self.quality_list = kwargs.pop("qlist", [90])
        self.repeat = repeat  
        if self.nocompress: print("Warning: no jpeg compression")
        self.in_memory = in_memory
        assert not self.in_memory # 读数据耗时占比似乎没那么高
        self.in_memory_cache = {}
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        # if self.demosaic:
        #     lines =list(filter(lambda x: "long" in x, lines))
        self.samples = []
        for line in lines:
            line = line.rstrip()
            self.samples.append(line.split(", "))  # "rgb_path, raw_path" or "raw_path"
        if ntest > 0 and "train" not in split:
            self.samples = self.samples[:ntest]
        if ntrain > 0 and "train" in split:
            self.samples = self.samples[:ntrain]
        self.transform = transform
        
    def get_format_and_fname(self, path):
        basename = os.path.basename(path)
        fname, format = basename.split(".")
        return fname, format

    def get_patch_folder_path(self, fpath):
        return os.path.join("datasets", "np_raw_patches", "_".join(fpath.removeprefix("./").split("/")))
    
    @staticmethod
    def rotate_and_resize(srgb, raw):
        ratio = ((srgb.shape[2] * raw.shape[3]) / (srgb.shape[3] * raw.shape[2]))
        ratio2 = ((srgb.shape[2] * srgb.shape[3]) / (raw.shape[3] * raw.shape[2]))
        if ratio > 1.1 or ratio<0.9:
            srgb = torch.rot90(srgb, 1,dims=[2,3]) # TODO: check if any image with 90
        # elif rotate not in [0,90]:
        #     print(raw_path, rotate)            
        if ratio2 > 3.8: # for raw image w/o demosaic
            srgb = F.interpolate(srgb, size=raw.shape[2:])
        if raw.shape[2:] != srgb.shape[2:]:
            boarder_h, boarder_w = raw.shape[2] - srgb.shape[2], raw.shape[3] - srgb.shape[3]
            h, w = srgb.shape[2:]
            if not (abs(boarder_h) < 40 and abs(boarder_w) < 40):
                # print(f"Size mismatch: {raw_path}, {srgb.shape}, {raw.shape}") 
                with open("datasets/fivek_dataset/delete_cases.txt", "a") as f:
                    f.write(raw_path+"\n")
                raise Exception(f"Size mismatch: {raw_path}, {srgb.shape}, {raw.shape}")
            else:
                assert boarder_h>0 and boarder_w>0
                raw = raw[:,:,boarder_h//2:boarder_h//2+h, boarder_w//2:boarder_w//2+w]
            # if resize:
            #     raw = F.interpolate(raw, size=srgb.shape[2:])
            # else:
            #     raise Exception(f"The shape between raw and srgb is not match {raw.shape} {srgb_image.shape}")
        return srgb, raw
        
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
        # get raw_path and jpeg_rgb_path
        index = index % len(self.samples)
        if len(self.samples[index]) == 1:
            raw_path = self.samples[index][0]
            rgb_path = None
        else:
            rgb_path, raw_path = self.samples[index]
            
        fname_raw, format_raw = self.get_format_and_fname(raw_path)
        
        if format_raw.lower() in ["png", "tif", "tiff"]:
            raw_img = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
            white_balance = -1
            raw_visible = raw_img
        else: # raw file
            raw_img = rawpy.imread(raw_path)
            white_balance = raw_img.camera_whitebalance
            if self.demosaic:
                if not self.only_demosaic:
                    raw_visible = raw_img.postprocess(output_color=rawpy.ColorSpace.XYZ, use_camera_wb=True, use_auto_wb=False, no_auto_bright=True, output_bps=16) #, gamma=(1,1)) # 
                else:
                    raw_visible = raw_img.postprocess(output_color=rawpy.ColorSpace.XYZ if self.xyz_space else rawpy.ColorSpace.raw, use_camera_wb=True, use_auto_wb=False, no_auto_bright=True, output_bps=16, gamma=(1,1))
        raw_visible = to_tensor(raw_visible/65535).float().unsqueeze(0)
        # try:
        #     srgb_image, raw_visible = rotate_and_resize(srgb_image, raw_visible)
        # except Exception as e:
        #     print(e)
        #     return 0
                
        jpg_bpp = 24          
        if len(self.samples[index]) == 1:
            srgb_image = raw_img.postprocess(
                use_camera_wb=True, half_size=False, no_auto_bright=True)
            format_rgb = "lossless"
            srgb_image = Image.fromarray(srgb_image)
        else:
            fname_rgb, format_rgb = self.get_format_and_fname(rgb_path)
            if format_rgb.lower() in ["jpg", "jpeg", "tif"]:
                srgb_image = Image.open(rgb_path) # .convert("RGB")
            else:
                raise NotImplementedError(
                    "The format %s is not implmented." % (fname_rgb))
        
        # if the RGB image is lossless (8-bit), then compress it using JPEG
        output = None
        if not self.nocompress:
            if not format_rgb.lower() in ["jpeg", "jpg"]:
                output = BytesIO()
                srgb_image.save(output, 'JPEG', quality=random.choice(self.quality_list) if "train" in self.split else self.quality_list[0])
                jpg_bpp = output.tell()*8/(srgb_image.size[1]*srgb_image.size[0])
                output.seek(0)
                srgb_image = Image.open(output)
            # with open("jpeg.jpg", "wb") as f:
            #     f.write(output.getvalue())                    

        srgb_image = srgb_image.convert("RGB")
        srgb_image = to_tensor(srgb_image).unsqueeze(0)

                # if srgb_image.shape[2:] != raw_visible.shape[2:]:
                #     boarder_h, boarder_w = raw_visible.shape[2] - srgb_image.shape[2], raw_visible.shape[3] - srgb_image.shape[3]
                #     h, w = srgb_image.shape[2:]
                #     raw_visible = raw_visible[:,:,boarder_h//2:boarder_h//2+h, boarder_w//2:boarder_w//2+w]
                # rotate = get_image_rotation(raw_path)

            # # Without demosaic
            # else:
            #     raw_visible = raw_img.raw_image_visible.astype(np.float32)
            #     # if raw_visible.s
            #     # lr_color_index = lr_raw.raw_colors_visible.astype(np.float32)
            #     assert np.std(raw_img.black_level_per_channel) < 2
            #     black_level = raw_img.black_level_per_channel[0]
            #     raw_visible = (raw_visible - black_level) / \
            #         (raw_img.white_level - black_level)
            #     if raw_visible.shape == srgb_image.shape[4:1:-1]:
            #         raw_visible = raw_visible.transpose(1, 0)
            #     # lr_tensor = torch.tensor(lr_visible)
            #     raw_visible = squeeze2d(raw_visible) # To make sure that have the fixed cmos pattern, e.g.g, RGGB
            #     srgb_image, raw_visible = rotate_and_resize(srgb_image, raw_visible)
            #     img_list = [srgb_image, raw_visible]
            #     if "train" in self.split:
            #         img_list[0] = squeeze2d(img_list[0])
            #     # print("save patches to file")
            #     # self.save_patch_to_folder(img_list, patch_folder_path)
            #     if not self.in_memory:
            #         raw_img.close()

        srgb_image, raw_visible = RawImageDataset.rotate_and_resize(srgb_image, raw_visible)
        img_list = [srgb_image, raw_visible]

        if output is not None:
            output.close()
            
        if self.transform is not None:
            img_list = self.transform(img_list)
        # else:
        #     if "train" in self.split:
        #         # H, W = raw_visible.shape[0], raw_visible.shape[1]
        #         img_list = random_crop(img_list, 1024)
        #         if not self.demosaic:
        #             img_list[0] = unsqueeze2d(img_list[0])

        for idx, img in enumerate(img_list):
            img_list[idx] = img.squeeze(0)

        return *img_list, {"wb": white_balance, "jpg_bpp": jpg_bpp, "raw_path": raw_path}
        # save_image(torch.stack(img_list, dim=0), "d.jpg")
    def __len__(self):
        return len(self.samples) * self.repeat




def find_misalignment(data):
    srgb = (resize(data[0],512).mean(dim=1).numpy()[0]*255).astype(np.uint8) # resize(data[0], 512)
    raw = (resize(data[1],512).mean(dim=1).numpy()[0]*255).astype(np.uint8)
    h, w = srgb.shape
    orb_detector = cv2.ORB_create(500)
    kp1, d1 = orb_detector.detectAndCompute(srgb, None)
    kp2, d2 = orb_detector.detectAndCompute(raw, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    sorted(matches, key=lambda x: x.distance, reverse=False)
    matches = matches[:10]
        # cv2.imwrite("d.jpg",cv2.drawMatches(srgb, kp1, raw, kp2, matches, None))
        
        # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt
    mean_dis = (points1-points2).mean(axis=0)
    if np.abs(mean_dis).min()>20:
            # print(mean_dis)
        cv2.imwrite(f"results/mismatch/{os.path.basename(data[2]['raw_path'][0]).split('.')[0]}.jpg",cv2.drawMatches(srgb, kp1, raw, kp2, matches, None))
        save_image(data[0], f"results/mismatch/{os.path.basename(data[2]['raw_path'][0]).split('.')[0]}_srgb.jpg")
        save_image(data[1], f"results/mismatch/{os.path.basename(data[2]['raw_path'][0]).split('.')[0]}_raw.jpg")
        # import pdb; pdb.set_trace()
    pass

if __name__ == "__main__":
    # dataset = RawImageDataset("./datasets/SID", split="val", transform=None, demosaic=True)
    # dataset = RawImageDataset("./datasets/fivek_dataset", split="val_Sony_jpg", transform=None, demosaic=True)
    # dataset = RawImageDataset("./datasets/fivek_dataset", split="train", demosaic=True, in_memory=True)
    dataset = RawImageDataset("./datasets/fivek_dataset", split="train_complicateISP", demosaic=True, ntest=-1)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, num_workers=16, batch_size=1)
    import tqdm
    import os
    from torchvision.transforms.functional import resize
    for data in tqdm.tqdm(dataloader):
        data
        pass
    
    
        # try:
        #     data = dataset[i]
        # except Exception as e:
        #     print(e)
    
    # import time

    # for i in range(15):
    #     ts = time.time()
    #     img_list = dataset[i]
    #     print(time.time()-ts)
        
        
    
    # # ts = time.time()
    # # img_list = dataset[0]
    # # print(time.time()-ts)
    # # ts = time.time()
    # # img_list = dataset[0]
    # # print(time.time()-ts)
    # pass
