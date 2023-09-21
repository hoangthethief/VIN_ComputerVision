import os
import numpy as np
import pandas as pd
from numpy.random import randint
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from PIL import Image



class LandscapeDataset(Dataset):
    def __init__(self, metadata_path, transform=None, test_mode=False):
        self.metadata = pd.read_csv(metadata_path)
        self.transform = transform
        self.test_mode = test_mode


    def __len__(self):
        return len(self.metadata)
    

    def __getitem__(self, idx):
        img_path = self.metadata.iloc[idx, 0]
        img = Image.open(img_path).convert('RGB')
        label = self.metadata.iloc[idx, 1]

        if self.transform:
            sample = self.transform(sample)

        return [img, label]
    

if __name__ == '__main__':
    metadata_path = 'data/metadata.csv'
    dataset = LandscapeDataset(metadata_path)

    # data_loader = DataLoader(dataset, batch_size=8, shuffle=None)
    c = 0
    for x, y in dataset:
        c += 1
        print(type(x), y)
        break