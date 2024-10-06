# Source: https://github.com/karpathy/nanoGPT
#
# MIT License
#
# Copyright (c) 2022 Andrej Karpathy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# The following code is modified from the BFN paper at https://github.com/nnaisense/bayesian-flow-networks
# Modifications:
# - Added data_adapters to GPT to preprocess the inputs and (optionally) postprocess the outputs
# - Added the `skip` option to concat the input and output of the network before the final projection
# - Added time `t` as an input to `forward()`

import math

import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

from ._base import register_model


def gelu(x):
    return F.gelu(x, approximate="tanh")


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, is_causal):
        super().__init__()
        assert n_embd % n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.is_causal = is_causal

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=self.is_causal
        )
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = F.softmax(att, dim=-1)
        # att = self.attn_dropout(att)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout, bias):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_head, n_embd, dropout, bias, is_causal):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = SelfAttention(n_head, n_embd, dropout, bias, is_causal)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def pe_encode(sequence_length: int, embedding_size: int) -> torch.Tensor:
    """Positional encoding as described in original attention is all you need paper"""
    pe = torch.zeros((sequence_length, embedding_size))
    pos = torch.arange(sequence_length).unsqueeze(1)
    pe[:, 0::2] = torch.sin(
        pos / torch.pow(1000, torch.arange(0, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    pe[:, 1::2] = torch.cos(
        pos / torch.pow(1000, torch.arange(1, embedding_size, 2, dtype=torch.float32) / embedding_size)
    )
    return pe


class TextInputAdapter(nn.Module):
    """
    A module to convert sequences of text class tokens to embedding tokens with learned positional embeddings.
    """

    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            output_size: int = 256,
    ):
        super().__init__()
        self.register_buffer("pos_embedding", pe_encode(seq_len, output_size))
        self.inp_embedding = nn.Linear(vocab_size, output_size)
        self.t_embedding = nn.Linear(1, output_size)

    def forward(self, probs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inp_emb = self.inp_embedding(2 * probs - 1)
        pos_emb = self.pos_embedding.unsqueeze(0).expand(inp_emb.size(0), -1, -1)
        t_emb = self.t_embedding((2 * t - 1).unsqueeze(-1)).unsqueeze(1)
        output = inp_emb + pos_emb + t_emb
        return output


@register_model("gpt")
class GPT(nn.Module):
    def __init__(
            self,
            vocab_size: int = 27,
            seq_len: int = 256,
            n_layer: int = 24,
            n_head: int = 12,
            n_embd: int = 768,
            dropout: float = 0.0,
            bias: bool = True,
            skip: bool = True,
            is_causal: bool = False,
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.input_adapter = TextInputAdapter(
            vocab_size=vocab_size,
            seq_len=seq_len,
            output_size=n_embd,
        )
        self.transformer = nn.ModuleDict(
            dict(
                drop=nn.Dropout(dropout),
                h=nn.ModuleList([Block(n_head, n_embd, dropout, bias, is_causal) for _ in range(n_layer)]),
                ln_f=LayerNorm(n_embd, bias=bias),
            )
        )
        self.is_causal = is_causal
        if self.is_causal:
            self.skip = False
        else:
            self.skip = skip
        if skip:
            self.lm_head = nn.Linear(2 * n_embd, vocab_size, bias=bias)
        else:
            self.lm_head = nn.Linear(n_embd, vocab_size, bias=bias)

        # init all weights
        self.apply(self._init_weights)

        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

        # report number of parameters
        # print(f"number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6:.2f}M")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, data: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = self.input_adapter(data, t)
        x = self.transformer.drop(x_in)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if self.skip:
            x = torch.cat([x, x_in], -1)
        logits = self.lm_head(x)
        return logits


class PositionalEncoding(nn.Module):
    """Positional encoding."""

    def __init__(self, num_hiddens, dropout=0., max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class TimeEmb(nn.Module):
    """
    https://github.com/huggingface/diffusers/blob/v0.11.0/src/diffusers/models/embeddings.py
    """

    def __init__(self, embed_dim, max_positions=2056):
        super().__init__()
        self.embedding_dim = embed_dim
        self.max_positions = max_positions

    def forward(self, timesteps, seq_len):
        # t: [B], mask: B,L
        # emb: B,L,D
        timesteps = timesteps * self.max_positions  # can be fractional
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_positions) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embedding_dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb[:, None, :].repeat(1, seq_len, 1)


@register_model('transformer')
class SeqTransformer(nn.Module):
    def __init__(self, ntoken: int, d_model=256, nhead=8, d_hid=256, nlayers=12, dropout=0.):
        super().__init__()
        self.d_model = d_model
        self.inp_embedding = nn.Linear(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.time_encoder = TimeEmb(d_model)
        self.embedding = nn.Embedding(ntoken + 2, d_model)
        self.feat_mixer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, batch_first=True, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.out = nn.Sequential(
            nn.Linear(d_hid, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, ntoken)
        )

    def forward(self, src, time):
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len]``
            time: Tensor, shape batch_size

        Returns:
            output Tensor of shape ``[batch_size, seq_len, ntoken]``
        """
        emb = self.inp_embedding(src)
        emb = self.feat_mixer(torch.cat([self.pos_encoder(emb), self.time_encoder(time, src.size(1))], dim=-1))
        output = self.transformer_encoder(emb)
        output = self.out(output)
        return output
