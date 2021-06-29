# from https://raw.githubusercontent.com/lucidrains/mlp-mixer-pytorch/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

import numpy as np
import util

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, int(dim * expansion_factor)),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(int(dim * expansion_factor), dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes),
        nn.LogSoftmax(dim=-1)
    )

# class HeadlessMLPMixer(nn.Module):
#     def __init__(self, len_a, len_b, depth, expansion_factor=4, dropout=0.):
#         super().__init__()
#         chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

#         self.input_shape = [len_a, len_b]
        
#         self.seq = nn.Sequential(
#             *[nn.Sequential(
#                 PreNormResidual(len_b, FeedForward(len_a, expansion_factor, dropout, chan_first)),
#                 PreNormResidual(len_b, FeedForward(len_b, expansion_factor, dropout, chan_last))
#             ) for _ in range(depth)],
#         )
#     def forward(self, x):
#         bs, is_ = util.bs_is_split(x.shape, np.prod(self.input_shape, dtype=int))
#         x = x.reshape(*bs, *self.input_shape)
#         x = self.seq(x)
#         x = x.reshape(*bs, *is_)
#         return x

