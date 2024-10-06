import torch
import numpy as np


def make_swiss_roll(n_samples=100, noise=1.0, pad=1.0, seed=None):
    """
    Make a swiss roll dataset on the 2-simplex (3 classes)
    :param n_samples: number of samples
    :param noise: noise scale
    :param pad: padding to avoid points on the simplex boundary
    :param seed: random seed
    :return:
        X: sampled points on the simplex, shape (n_samples, 3)
        t: time steps of the points, useful for plotting, shape (n_samples,)
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    t = 1.5 * np.pi * (1 + 2 * torch.rand(n_samples, generator=generator))
    x = t * torch.cos(t)
    y = t * torch.sin(t)

    x += noise * torch.randn(n_samples, dtype=torch.float, generator=generator)
    y += noise * torch.randn(n_samples, dtype=torch.float, generator=generator)
    # shift x and y
    x = x - x.min() + pad
    y = y - y.min() + pad
    total_max = (x + y).max() + pad
    z = total_max - (x + y)
    # normalize points on the simplex
    X = torch.stack([x, y, z], dim=1) / total_max
    return X, t
