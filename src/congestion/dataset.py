import numpy as np
import torch
from torch.utils.data import Dataset


class CongestionWindowDataset(Dataset):
    """Wraps pre-computed sliding windows (see preprocess.py) as a torch Dataset."""

    def __init__(self, X, y_volume, y_level):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_volume = torch.from_numpy(y_volume.astype(np.float32))
        self.y_level = torch.from_numpy(y_level.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_volume[idx], self.y_level[idx]


def load_split(npz_path, split):
    data = np.load(npz_path, allow_pickle=True)
    return CongestionWindowDataset(data[f"X_{split}"], data[f"y_{split}"], data[f"lvl_{split}"])
