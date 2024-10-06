import os
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

from ._base import register_dataset


@register_dataset('bmnist')
class BinaryMNIST(Dataset):
    """
    Binarized MNIST dataset.
    """
    data_url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'

    def __init__(self, root, split, transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.transform = transform

        data_path = os.path.join(self.root, f'binarized_mnist_{split}.amat')
        if not os.path.exists(self.root):
            os.makedirs(self.root, exist_ok=True)
        if not os.path.exists(data_path):
            print(f'Downloading {split} set...')
            urlretrieve(self.data_url.format(split), data_path)
        self.data = torch.from_numpy(np.loadtxt(data_path).astype(np.float32))

    def __getitem__(self, index):
        x = self.data[index]
        x = torch.stack([x, 1 - x], dim=-1)
        if self.transform is not None:
            x = self.transform(x)
        return (x,)

    def __len__(self):
        return self.data.size(0)
