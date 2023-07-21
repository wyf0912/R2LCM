from compressai.datasets.raw_image import RawImageDataset


def down4_and_save(data):
    h, w = data[0].shape[2:]
    srgb = resize(data[0], [h//4, w//4], interpolation=InterpolationMode.BICUBIC)
    raw = resize(data[1], [h//4, w//4], interpolation=InterpolationMode.BICUBIC)
    fname = os.path.basename(data[2]["raw_path"][0]).split(".")[0]
    srgb_path = f"datasets/fivek_dataset/export_down4/{fname}.png"
    raw_path = f"datasets/fivek_dataset/raw_down4/{fname}.tif"
    save_image(srgb, srgb_path)
    raw_uint16 = (raw.numpy()*65535).clip(0,65535).astype("uint16")
    tiff.imsave(raw_path, raw_uint16[0])

if __name__ == "__main__":
    # dataset = RawImageDataset("./datasets/SID", split="val", transform=None, demosaic=True)
    # dataset = RawImageDataset("./datasets/fivek_dataset", split="val_Sony_jpg", transform=None, demosaic=True)
    # dataset = RawImageDataset("./datasets/fivek_dataset", split="train", demosaic=True, in_memory=True)
    dataset = RawImageDataset("./datasets/fivek_dataset", split="val_complicateISP", demosaic=True, ntest=-1)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, num_workers=16, batch_size=1)
    import tqdm
    import os
    from torchvision.transforms.functional import resize, InterpolationMode
    from skimage import io
    from torchvision.utils import save_image
    import tifffile as tiff
    import multiprocessing
    import threading
    from concurrent.futures import ThreadPoolExecutor
    
    # for data in tqdm.tqdm(dataloader):
    #     down4_and_save(data)    
        
    pool = ThreadPoolExecutor(max_workers=5)
    for data in tqdm.tqdm(dataloader):
        # pool.apply_async(down4_and_save, args=(data,))
        # down4_and_save(data)
        pool.submit(down4_and_save, data)

    