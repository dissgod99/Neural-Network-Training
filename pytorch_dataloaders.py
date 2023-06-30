import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import math

DATA_PATH = "data/wine_dataset.csv"
BATCH_SIZE = 4
EPOCHS = 2

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

dataloader = DataLoader(dataset=dataset,
                        batch_size=4,
                        shuffle=True)

data_iterator = iter(dataloader)

# training loop
total_samples = dataset.__len__()
n_iterations = math.ceil(total_samples/BATCH_SIZE)

print(f"Total Samples: {total_samples}")
print(f"Iterations: {n_iterations}")

for epoch in range(EPOCHS):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update weights
        if (i+1) % 5 == 0:
            print(f"epoch {epoch+1}/{EPOCHS}, step {i+1}/{n_iterations}, inputs {inputs.shape}")



