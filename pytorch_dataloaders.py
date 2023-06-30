import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import math

DATA_PATH = "data/wine_dataset.csv"


class WineDataset(Dataset):

    def __init__(self):
        #load data
        self.xy = np.loadtxt(DATA_PATH, delimiter=",", skiprows=1, dtype=np.float32)
        self.x = torch.from_numpy(self.xy[:, 1:])
        self.y = torch.from_numpy(self.xy[:, [0]]) #shape= (n_samples, 1)
        self.n_samples = self.xy.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


dataset = WineDataset()
first_data = dataset[0]

print(first_data[0].numpy())


