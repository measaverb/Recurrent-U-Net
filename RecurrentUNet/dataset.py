import os
import pandas as pd
import numpy as np
import random
from cv2 import cv2

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToPILImage
from torchvision import transforms, utils


class Vein(Dataset):
    def __init__(self, root='./data/', split='train', transform=None, train_p=70):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform
        self.items = []
        self.train_p = train_p

        folder_list = os.listdir('./data')
        
        for folder in folder_list:
            self.items.extend(os.listdir('./data/'+folder))

        random.shuffle(self.items)
        self.train_list = []
        for i in range(int(0.7 * len(self.items))):
            self.train_list.append(self.items[i])

        self.items = list(set(self.items) - set(self.train_list))
        self.test_list = self.items
        
    def __getitem__(self, index):
        if self.split == 'train':
            src = cv2.imread('./data/image/'+self.train_list[index])
            mask = cv2.imread('./data/mask/'+self.train_list[index])
        else:
            src = cv2.imread('./data/image/'+self.test_list[index])
            mask = cv2.imread('./data/mask/'+self.test_list[index])
        sample = (src, mask)

        return sample if self.transform is None else self.transform(*sample)

    def __len__(self):
        if self.split == 'train':
            return len(self.train_list)
        else:
            return len(self.test_list)