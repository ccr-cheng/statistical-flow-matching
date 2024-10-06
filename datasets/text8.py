import os
import requests
import zipfile
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

from ._base import register_dataset


def prepare_text8(root):
    os.makedirs(root, exist_ok=True)
    zip_fname = os.path.join(root, 'text8.zip')
    if not os.path.exists(zip_fname):
        data_url = 'http://mattmahoney.net/dc/text8.zip'
        with open(zip_fname, 'wb') as f:
            print('Downloading text8')
            f.write(requests.get(data_url).content)
            print('Done!')
        with zipfile.ZipFile(zip_fname) as f:
            f.extractall(root)
        os.remove(zip_fname)

    with open(os.path.join(root, 'text8'), 'r') as f:
        data = f.read()
    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print('all the unique characters:', ''.join(chars))
    print(f'vocab size: {vocab_size:,}')

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # encode both to integers
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9): int(n * 0.95)]
    test_data = data[int(n * 0.95):]
    train_ids = encode(train_data)
    val_ids = encode(val_data)
    test_ids = encode(test_data)
    print(f'train has {len(train_ids):,} tokens')
    print(f'val has {len(val_ids):,} tokens')
    print(f'test has {len(test_ids):,} tokens')

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    test_ids = np.array(test_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(root, 'train.bin'))
    val_ids.tofile(os.path.join(root, 'valid.bin'))
    test_ids.tofile(os.path.join(root, 'test.bin'))
    print(f'Saved to dir {root}')

    # save the meta information as well, to help us encode/decode later
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(os.path.join(root, 'meta.pkl')), 'wb') as f:
        pickle.dump(meta, f)
    print(f'text8 dataset downloaded and prepared in dir {root}')


@register_dataset('text8')
class Text8Dataset(Dataset):
    def __init__(self, root, split, vocab_size=27, seq_len=256):
        """
        seq_len should include context length. Example: seq_len=512 for modeling 256 chars with 256 char of context.
        context is only used for correct preparation of val/test sets.
        """
        self.root = root
        self.split = split
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        fname = os.path.join(root, f'{split}.bin')
        if not os.path.exists(fname):
            prepare_text8(root)
        self.data = np.memmap(fname, np.uint16, 'r')

    def __getitem__(self, index):
        seq = torch.from_numpy(self.data[index: index + self.seq_len].astype(np.int64))
        seq = F.one_hot(seq, self.vocab_size).float()
        return (seq,)

    def __len__(self):
        return self.data.size - self.seq_len
