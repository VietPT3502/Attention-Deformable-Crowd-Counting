import math
import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2
import glob
import scipy.io as io
import matplotlib.pyplot as plt
from albumentations.pytorch.transforms import ToTensorV2
import scipy
import scipy.spatial
import scipy.ndimage
import h5py
class DroneCrowd(Dataset):
    def __init__(self, data_root, gt_downsample, transform=None, type="train"):
        self.root_path = data_root
        self.gt_downsample = gt_downsample
        if type == "train":
            self.set_dir = os.path.join(data_root, "train")
        elif type == "val":
            self.set_dir = os.path.join(data_root, "val")
        elif type == "test":
            self.set_dir = os.path.join(data_root, "test")
        else:
            assert "?"
        self.images_dir = [os.path.join(self.set_dir, "images", item) for
                          item in sorted(os.listdir(os.path.join(self.set_dir,  "images")))]
        self.annotations_dir = [os.path.join(self.set_dir, "Annotations_h5", item) for
                               item in sorted(os.listdir(os.path.join(self.set_dir, "Annotations_h5")))]
        self.transform = transform
        self.type = type 

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, index):
        # assert index >= len(self), 'index range error'
        if index >= len(self):
            print("error index", index)

        # load image and ground truth
        img, den1= self.load_data(index)

        if self.type == "train":  
            crop_factor = 0.5
            crop_size = (int(img.size[0]*crop_factor),int(img.size[1]*crop_factor))
            dx = int(random.random()*img.size[0]*crop_size[0] / img.size[0])
            dy = int(random.random()*img.size[1]*crop_size[1] / img.size[1])
            img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
            den1 = den1[int(dy/2):int(crop_size[1]/2+dy/2),int(dx/2):int(crop_size[0]/2+dx/2)]

        
        den2 = cv2.resize(den1,(int(math.ceil(den1.shape[1]/2)),int(math.ceil(den1.shape[0]/2))),interpolation = cv2.INTER_CUBIC)*4
        den3 = cv2.resize(den1,(int(math.ceil(den1.shape[1]/4)),int(math.ceil(den1.shape[0]/4))),interpolation = cv2.INTER_CUBIC)*16
        
        img = np.asarray(img)
        # gt_map = torch.from_numpy(gt_map).unsqueeze(0)
        # img = img.transpose((1, 2, 0))
        # applu augumentation
        input_dict = {
        'image': img,
        'den1': den1,
        'den2': den2,
        'den3': den3
    }
        if self.transform is not None:
            augmented = self.transform(**input_dict)
            img = augmented['image']
            den1 = augmented['den1']
            den2 = augmented['den2']
            den3 = augmented['den3']


        return img, den1, den2, den3


    def load_data(self, index):
        img_path = self.images_dir[index]
        img = Image.open(img_path).convert('RGB')
        gt_path = self.annotations_dir[index]
        gt_file = h5py.File(gt_path, "r")
        den = np.asarray(gt_file['density'])

        return img, den

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0), auto=True, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image while meeting stride-multiple constraints
    Returns:
    im (array): (height, width, 3)
    ratio (array): [w_ratio, h_ratio]
    (dw, dh) (array): [w_padding h_padding]
    """
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):  # [h_rect, w_rect]
         new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # wh ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # w h
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])  # [w h]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # [w_ratio, h_ratio]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)