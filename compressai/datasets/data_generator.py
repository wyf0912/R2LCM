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

import glob
import cv2
import numpy as np

aug_times = 1

def data_aug(img, mode=0):

    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(img)
    elif mode == 2:
        return np.rot90(img)
    elif mode == 3:
        return np.flipud(np.rot90(img))
    elif mode == 4:
        return np.rot90(img, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        return np.rot90(img, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_name,patch_size=48,stride=48):

    # read image
    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16 
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    h, w, cc = img.shape
    if h > w:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w, cc = img.shape
    patches = []
    # extract patches
    for i in range(0, h-patch_size+1, stride):
        for j in range(0, w-patch_size+1, stride):
            x = img[i:i+patch_size, j:j+patch_size,:]
            # data aug
            for k in range(0, aug_times):
                x_aug = data_aug(x, mode=0)
                patches.append(x_aug)
            
    return patches

def datagenerator(data_dir, batch_size=128, patch_size=48, stride=48, verbose=False, ftype='png', debug=False):
    
    file_list = sorted(glob.glob(data_dir+'/*.{}'.format(ftype)))  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    # for i in range(len(file_list)):
    length = 10 if debug else len(file_list)
    for i in range(length):
        patch = gen_patches(file_list[i], patch_size=patch_size, stride=stride)
        data.append(patch)        
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done')
    data = np.concatenate(data)#, dtype='uint8')
#    data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    discard_n = len(data)-len(data)//batch_size*batch_size
    data = np.delete(data,range(discard_n),axis = 0)
    print('^_^-training data finished-^_^')
    return data

def get_image(file_name):

    # read image
    img = cv2.imread(file_name,cv2.IMREAD_UNCHANGED)  # -1 => any depth, in this case 16 
    img = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.float32)
    h, w, _ = img.shape
    h_ = h - h % 16
    w_ = w - w % 16
    h__ = (h % 16) // 2
    w__ = (w % 16) // 2
    img = img[h__:h__+h_, w__:w__+w_]
    return img

def datagenerator_test(data_dir, verbose=False, ftype='png'):
    
    file_list = sorted(glob.glob(data_dir+'/*.{}'.format(ftype)))  # get name list of all .png files
    # initialize
    data = []
    # generate patches
    for i in range(len(file_list)):
        patch = get_image(file_list[i])
        data.append(patch)        
        if verbose:
            print(str(i+1)+'/'+ str(len(file_list)) + ' is done')
    print('^_^-training data finished-^_^')
    return data