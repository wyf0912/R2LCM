import os
import glob
import random
import itertools


def generate_flist(root):
    devices = ["SonyA57", "SamsungNX2000", "OlympusEPL6"]
    
    for device in devices:
        for phase in ["train", "val", "test"]:
            raw_list = glob.glob(f"{root}/{device}/{phase}/raw/*.tif") + glob.glob(f"{root}/{device}/{phase}/raw/*.png")
            srgb_list = glob.glob(f"{root}/{device}/{phase}/sRGB/*.tif") + glob.glob(f"{root}/{device}/{phase}/sRGB/*.png")
            raw_list, srgb_list = sorted(raw_list), sorted(srgb_list)
            assert len(srgb_list) == len(raw_list)
            with open(f"./datasets/NUS/cvpr22_{device}_{phase}.txt", "w") as f:
                for srgb_path, raw_path in zip(srgb_list, raw_list):
                    f.write(f"{srgb_path}, {raw_path}\n")
if __name__ == "__main__":
    generate_flist("/home/Dataset/DatasetYufei/content_aware_reconstruction")
