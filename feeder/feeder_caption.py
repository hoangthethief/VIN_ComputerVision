import os
import numpy as np
import pandas as pd
from numpy.random import randint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import spacy



class LandscapeDataset(Dataset):
    def __init__(self, metadata_path, test_mode=False):
        self.text_embedder = spacy.load('en_core_web_lg')

        self.metadata = pd.read_csv(metadata_path)
        self.test_mode = test_mode

        self.label = ['Newyork', 'Singapore', 'Sydney', 'Venezia', 'Amsterdam', 'Roma',
                        'Moscow', 'Hanoi', 'Rabat', 'Kyoto', 'Dubai', 'Rio', 'Maldives',
                        'Paris', 'London']
        self.mapping = {self.label[i]: i for i in range(len(self.label))}


        self.transform = A.Compose([
                                A.SmallestMaxSize(max_size=360),
                                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                                A.RandomCrop(height=256, width=256),
                                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                                A.RandomBrightnessContrast(p=0.5),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ToTensorV2()
                                ])

        if test_mode:
            self.transform = A.Compose([
                                    A.SmallestMaxSize(max_size=360),
                                    A.CenterCrop(height=256, width=256),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                    ToTensorV2()])



    def __len__(self):
        return len(self.metadata)
    

    def __getitem__(self, idx):
        img_path = 'data/' + self.metadata.iloc[idx, 0]
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = self.metadata.iloc[idx, 1]
        label = self.mapping[label]

        caption = self.metadata.iloc[idx, 2]
        caption = self.text_embedder(caption).vector

        img = self.transform(image=img)['image']

        return [img_path, img, caption, label]
    
    

if __name__ == '__main__':
    metadata_path = 'feeder/train.csv'
    dataset = LandscapeDataset(metadata_path)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    c = 0

    for path, image, caption, y in data_loader:
        print(path)
        print(image.shape)
        print(caption.shape)
        print(y)
        print("======")

        break
