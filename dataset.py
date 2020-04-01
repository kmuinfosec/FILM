import pickle

import torch
from torch.utils.data import Dataset

class PklDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __data_generation(self, fn_list):
        vector_array = []
        for fn in fn_list:
            with open(fn, 'rb') as f:
                vector_array.append(pickle.load(f))
        return vector_array

    def __getitem__(self, idx):
        x = self.__data_generation(self.x[idx])
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(self.y[idx])

