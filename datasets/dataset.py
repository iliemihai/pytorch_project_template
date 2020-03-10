import torch
import torchtext
import tokenizers
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    def __init__(self, data):
        """Initialization"""
        self.labels = [row[1] for row in data]
        self.list_IDs = [row[0] for row in data]

    def __len__(self):
        """Denotes total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates the sample of data"""

        # Load data and get label
        X = self.list_IDs[index]
        y = self.labels[index]

        return X, y
