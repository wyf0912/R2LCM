"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""

import os
import cv2
import numpy as np

import torch

from compressai.datasets import data_generator as dg 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DatasetRAW(object):
    def __init__(self, root, batch_size, patch_size, stride, gamma_flag = True, to_gpu=True, ftype='png', debug=False):
        self.root = root
        self.gamma_flag = gamma_flag
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.stride = stride

        # load all image files
        rawimgs=[]
        srgbimgs=[]
        
        print('Inside train_datagen \n')
        rawimgs = dg.datagenerator(os.path.join(self.root,'raw'), batch_size=batch_size, patch_size=patch_size, stride=stride, ftype=ftype, debug=debug)
        print(rawimgs.shape[0])
        print('Outside train_datagen \n')
        rawimgs = torch.from_numpy(rawimgs.astype(np.float32))
        rawimgs = rawimgs.permute(0, 3, 1, 2)
        rawimgs = rawimgs / 65535.0
        if to_gpu:
            rawimgs = rawimgs.to(device)
        self.rawimgs = rawimgs
        
        print('Inside train_datagen \n')
        srgbimgs = dg.datagenerator(os.path.join(self.root,'sRGB'), batch_size=batch_size, patch_size=patch_size, stride=stride, ftype=ftype, debug=debug)
        print(srgbimgs.shape[0])
        print('Outside train_datagen \n')
        srgbimgs = torch.from_numpy(srgbimgs.astype(np.float32))
        srgbimgs = srgbimgs.permute(0, 3, 1, 2)
        # The max value differs according to dataset (png or tif)
        if ftype == 'png':
            srgbimgs = srgbimgs / 255.0
        elif ftype == 'tif':
            srgbimgs = torch.round(srgbimgs/255) / 255.0
            # srgbimgs = srgbimgs/65535.0
        else:
            raise ValueError('ftype is not valid.')
        if to_gpu:
            srgbimgs = srgbimgs.to(device)
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images
        target = self.rawimgs[idx]
        img = self.srgbimgs[idx]
        return img, target, {"wb": -1, "jpg_bpp": 24, "raw_path": ""}

    def __len__(self):
        return len(self.rawimgs)


class DatasetRAWTest(object):
    def __init__(self, root, to_gpu=True, ftype='png'):
        self.root = root
        self.to_gpu = to_gpu
        self.ftype = ftype
        # load all image files
        rawimgs=[]
        srgbimgs=[]
        
        print('Inside test_datagen \n')
        rawimgs = dg.datagenerator_test(os.path.join(self.root,'raw'), ftype=ftype)
        print(len(rawimgs))
        print('Outside test_datagen \n')
        
        self.rawimgs = rawimgs
        
        print('Inside test_datagen \n')
        srgbimgs = dg.datagenerator_test(os.path.join(self.root,'sRGB'), ftype=ftype)
        print(len(srgbimgs))
        print('Outside test_datagen \n')
        self.srgbimgs = srgbimgs

    def __getitem__(self, idx):
        # load images
        target = self.rawimgs[idx]
        target = torch.from_numpy(target.astype(np.float32))
        target = target.permute(2, 0, 1)
        target = target / 65535.0
        if self.to_gpu:
            target = target.to(device)

        img = self.srgbimgs[idx]
        if img.shape[0] != self.rawimgs[idx].shape[0]:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = torch.from_numpy(img.astype(np.float32))
        img = img.permute(2, 0, 1)
        # The max value differs according to dataset (png or tif)
        if self.ftype == 'png':
            img = img/255.0
        elif self.ftype == 'tif':
            # img = img/65535.0
            img = torch.round(img/255)/255.0
        if self.to_gpu:
            img = img.to(device)

        return img, target, {"wb": -1, "jpg_bpp": 24, "raw_path": ""}

    def __len__(self):
        return len(self.rawimgs)