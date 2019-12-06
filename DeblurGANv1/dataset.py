import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size
        # Start from h1 and w1
        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)
        # Crop h1 ~ h2 and w1 ~ w2
        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw
        
    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1 : self.h2, self.w1 : self.w2, :]
        else:
            return img[self.h1 : self.h2, self.w1 : self.w2]

class DeblurDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.imglist_A = utils.get_files(opt.baseroot_A).sort()
        self.imglist_B = utils.get_files(opt.baseroot_B).sort()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # Blurry image
        img = Image.open(self.imglist_A[index]).convert('RGB')
        cropper = RandomCrop(img.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        img = cropper(img)
        img = self.transform(img)
        # Clean image
        target = Image.open(self.imglist_B[index]).convert('RGB')
        cropper = RandomCrop(target.shape[:2], (self.opt.crop_size, self.opt.crop_size))
        target = cropper(target)
        target = self.transform(target)
        return img, target
    
    def __len__(self):
        return len(self.imglist_A)
