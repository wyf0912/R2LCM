from posixpath import join
import torch.utils.data as data
import numpy as np
import pickle
import lmdb
from io import BytesIO
import random
from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
import torch

class LMDBDataset(data.Dataset):
    def __init__(self, db_path_srgb, db_path_raw, nocompress, size=None, repeat=1, qlist=None):
        self.nocompress = nocompress
        if self.nocompress: 
            print("Warning: no jpeg compression")
        else:
            assert qlist is not None 

        self.quality_list = qlist
        self.db_path_srgb = db_path_srgb
        self.db_path_raw = db_path_raw
        
        self.env_srgb = lmdb.open(db_path_srgb, max_readers=10, readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.env_raw = lmdb.open(db_path_raw, max_readers=10, readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env_srgb.begin(write=False) as txn:
            length = txn.stat()['entries']
            
        self.length = size or length
        self.repeat = repeat
        self.meta = pickle.load(open(join(self.db_path_raw, 'meta_info.pkl'), 'rb'))
        self.shape = self.meta['shape']
        self.dtype = self.meta['dtype']

    def __getitem__(self, index):
        env_srgb = self.env_srgb
        env_raw = self.env_raw
        index = index % self.length
        
        with env_srgb.begin(write=False) as txn:
            srgb_data = txn.get('{:08}'.format(index).encode('ascii'))
            
        with env_raw.begin(write=False) as txn:
            raw_data = txn.get('{:08}'.format(index).encode('ascii'))
            wb = self.meta[index]["wb"]
        srgb_patch = np.frombuffer(srgb_data, np.uint8)
        raw_patch = np.frombuffer(raw_data, self.dtype)
        srgb_patch = srgb_patch.reshape(*self.shape)
        raw_patch = raw_patch.reshape(*self.shape)
    
        jpg_bpp = 24
        if not self.nocompress:
            output = BytesIO()
            srgb_image = Image.fromarray(srgb_patch.transpose([1,2,0]))
            srgb_image.save(output, 'JPEG', quality=random.choice(self.quality_list))
            jpg_bpp = output.tell()*8/(srgb_image.size[1]*srgb_image.size[0])
            output.seek(0)
            srgb_image = srgb_image.convert("RGB")
            srgb_patch = np.array(Image.open(output)).transpose(2,0,1)
            
        img_list = [torch.tensor(srgb_patch/255., dtype=torch.float32), torch.tensor(raw_patch/255./255., dtype=torch.float32)]
        return *img_list, {"wb": wb, "jpg_bpp": jpg_bpp}

    def __len__(self):
        return int(self.length * self.repeat)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'
    
    
if __name__ == "__main__":
    dataset = LMDBDataset(db_path_srgb="datasets/LMDB_Train/nus_srgb.db", db_path_raw="datasets/LMDB_Train/nus_raw_onlyDemosaic.db", nocompress=True)
    import tqdm
    for i in tqdm.tqdm(range(len(dataset))):
        try:
            data = dataset[i]
        except Exception as e:
            print(e)
        
    # save_image(list(data[:2]), "d.jpg")
    
    # dataset = LMDBDataset(db_path_srgb="datasets/LMDB_Train/nus_srgb.db", db_path_raw="datasets/LMDB_Train/nus_raw_onlyDemosaic.db", nocompress=False, qlist=[10])
    # data = dataset[100]
    # save_image(list(data[:2]), "d2.jpg")
    