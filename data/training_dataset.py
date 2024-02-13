import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import cv2


class TrainingDataset(Dataset):

    def __init__(self, data_path, split_size=256, patch_size=384, transform=None, load_data=True):
        super(TrainingDataset, self).__init__()
        self.imgs_paths = list((Path(data_path) / f'imgs_{patch_size}').glob('*'))
        self.gt_imgs_paths = [img.parent.parent / ('gt_' + img.parent.name) / img.name for img in self.imgs_paths]

        self.load_data = load_data
        if self.load_data:
            self.imgs = [cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB) for img_path in self.imgs_paths]
            self.gt_imgs = [np.array(Image.open(gt_img_path).convert('L')) for gt_img_path in self.gt_imgs_paths]
        else:
            self.imgs = self.imgs_paths
            self.gt_imgs = self.gt_imgs_paths

        self.split_size = split_size
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.load_data:
            sample = self.imgs[index]
            gt_sample = self.gt_imgs[index]
        else:
            sample = Image.open(self.imgs[index]).convert("RGB")
            gt_sample = Image.open(self.gt_imgs[index]).convert("L")

        if self.transform:
            transform = self.transform({'image': sample, 'gt': gt_sample, 'img_path': self.imgs_paths[index], 'gt_img_path': self.gt_imgs_paths[index]})
            sample = transform['image']
            gt_sample = transform['gt']

        gt_sample = gt_sample.float()
        return sample, gt_sample
