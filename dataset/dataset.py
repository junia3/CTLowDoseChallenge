from typing import Iterable
import torch
import pandas as pd
import numpy as np
import math
import cv2, os
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import pdb

def get_patch(full_input_img, full_target_img, patch_n, patch_size):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)

def normalize(image):
    normalized = (image - image.min())/(image.max()-image.min())
    return normalized

class CTDataset(Dataset):
    def __init__(self, mode="train", transform=None, patch=False):
        self.transform = transform
        self.directory = f"/data/ct/dataset/{mode}_dataset"
        self.patch = patch

    def __len__(self):
        return len(os.listdir(self.directory))

    def __getitem__(self, idx):
        with open(os.path.join(self.directory, f"{str(idx).zfill(4)}.npy"), "rb") as f:
            source = np.load(f)
            target = np.load(f)
        
        if not self.patch:
            source, target = torch.FloatTensor(source), torch.FloatTensor(target)

            if self.transform is not None:
                mix = torch.stack([source, target])
                mix = self.transform(mix)
                source, target = mix[0], mix[1]

            source, target = normalize(source).unsqueeze(0), normalize(target).unsqueeze(0)
            
        else:
            input_patches, target_patches = get_patch(source, target, 10, 64)
            source = torch.FloatTensor(input_patches)
            target = torch.FloatTensor(target_patches)
            
            if self.transform is not None:
                mix = torch.stack([source, target])
                mix = self.transform(mix)
                source, target = mix[0], mix[1]

            source, target = normalize(source), normalize(target)
            
        return source.float(), target.float()

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation((-180, 180)),
        transforms.RandomAffine((-180, 180), (0, 0.01), scale=(0.9, 1.1), shear=(-2, 2)),
    ])

    test = CTDataset("test", None, True)
    source, target = test[0]