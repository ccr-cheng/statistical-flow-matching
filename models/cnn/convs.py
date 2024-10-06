import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_layers import ResidualBlock, RefineBlock, get_act
from .normalization import get_normalization
from .._base import register_model


def _compute_cond_module(module, x):
    for m in module:
        x = m(x)
    return x


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = np.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


@register_model('cnn')
class ConvNet(nn.Module):
    def __init__(self, in_channels=2, normalization='InstanceNorm++', ngf=128, nonlinearity='elu'):
        super().__init__()
        self.in_channels = in_channels
        self.normalization = normalization
        self.ngf = ngf
        self.nonlinearity = nonlinearity

        self.norm = get_normalization(normalization, conditional=False)
        self.act = act = get_act(nonlinearity)

        self.begin_conv = nn.Conv2d(in_channels, ngf, 3, stride=1, padding=1)
        self.fc_t1 = nn.Linear(ngf, ngf)
        self.normalizer = self.norm(ngf, 0)
        self.end_conv = nn.Conv2d(ngf, in_channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act, normalization=self.norm)
        ])
        self.fc_t2 = nn.Linear(ngf, ngf)

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act, normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act, normalization=self.norm)
        ])
        self.fc_t3 = nn.Linear(ngf, 2 * ngf)

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)
        ])
        self.fc_t4 = nn.Linear(ngf, 2 * ngf)

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm, adjust_padding=False, dilation=4),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)
        ])
        self.fc_t5 = nn.Linear(ngf, 2 * ngf)

        self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def forward(self, x, t):
        x = x.permute(0, 3, 1, 2)
        t_embed = get_time_embedding(t, self.ngf)
        output = self.begin_conv(2 * x - 1) + self.fc_t1(t_embed)[..., None, None]

        layer1 = _compute_cond_module(self.res1, output) + self.fc_t2(t_embed)[..., None, None]
        layer2 = _compute_cond_module(self.res2, layer1) + self.fc_t3(t_embed)[..., None, None]
        layer3 = _compute_cond_module(self.res3, layer2) + self.fc_t4(t_embed)[..., None, None]
        layer4 = _compute_cond_module(self.res4, layer3) + self.fc_t5(t_embed)[..., None, None]

        ref1 = self.refine1([layer4], layer4.shape[2:])
        ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
        ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
        output = self.refine4([layer1, ref3], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)
        return output.permute(0, 2, 3, 1)
