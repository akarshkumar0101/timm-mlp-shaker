import numpy as np
import torch
from torch import nn

from einops.layers.torch import Rearrange, Reduce

import copy

import dim_models
import util

def FeedForward(shape, dim_to_mix, out_features=None, 
                bias=True, expansion_factor=4, dropout=0.):
    in_features = shape[dim_to_mix]
    if out_features is None:
        out_features = in_features
    n_hidden = int(in_features*expansion_factor)
    shape1 = copy.copy(list(shape))
    shape2 = copy.copy(shape1)
    shape2[dim_to_mix] = n_hidden
    return nn.Sequential(
        dim_models.DimLinear(shape1, dim_to_mix, out_features=n_hidden, bias=bias),
        nn.GELU(),
        nn.Dropout(dropout),
        dim_models.DimLinear(shape2, dim_to_mix, out_features=out_features, bias=bias),
        nn.Dropout(dropout)
    )
    
class PreNormResidual(nn.Module):
    def __init__(self, shape, fn, dims_to_mix=[-1], normalize=True, residual=True):
        super().__init__()
        self.normalize = normalize
        self.residual = residual
        
        self.fn = fn
        self.norm = dim_models.DimLayerNorm(shape, dims_to_mix) if self.normalize else None

    def forward(self, x):
        nx = self.norm(x) if self.normalize else x
        y = self.fn(nx)
        return y+x if self.residual else y
        
class MLPFlatShaker(nn.Module):
    """
    Maps from (bs, N) to (bs, N)
    where N = flat_prod(shape).
    
    This reshapes N->*dims (for example 100->2,10,5), 
    then applies linear layers to different dims of this new shape.
    The dims_to_mix determines which dimensions are mixed in which order.
    """
    def __init__(self, shape, dims_to_mix, target_lengths=None, 
                 expansion_factor=4, dropout=0., normalize=True, residual=True, verbose=False):
        super().__init__()
        
        self.input_shape = shape
        self.ndims = len(self.input_shape)
        self.dims_from_left = [(d if d>=0 else d+self.ndims) for d in dims_to_mix]
        self.dims_from_right = [d-self.ndims for d in self.dims_from_left]
        
        if target_lengths is None:
            target_lengths = [None]*len(self.dims_from_left)
        self.target_lengths = target_lengths
        
        fshape = copy.copy(list(self.input_shape)) # forward shape (as its changing)
        self.mix_dims = nn.ModuleList([])
        for dim, target_length in zip(self.dims_from_left, self.target_lengths):
            if target_length is None:
                target_length = fshape[dim]
                
            m = FeedForward(copy.copy(fshape), dim_to_mix=dim, out_features=target_length,
                            expansion_factor=expansion_factor, dropout=dropout)
            m = PreNormResidual(fshape, m, dims_to_mix=-1, normalize=normalize, residual=residual)
            self.mix_dims.append(m)
            fshape[dim] = target_length
            
        self.output_shape = copy.copy(fshape)
        
    def forward(self, x):
        bs, is_ = util.bs_is_split(x.shape, np.prod(self.input_shape, dtype=int))
        x = x.reshape(*bs, *self.input_shape)
        for i, m in enumerate(self.mix_dims):
            x = m(x)
        x = x.reshape(*bs, *is_)
        return x
    
def ViMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    shape = [num_patches, dim]

    return nn.Sequential(
        Rearrange('b c (nph psh) (npw psw) -> b (nph npw) (psh psw c)', psh=patch_size, psw=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        MLPFlatShaker(shape, [0, 1]*depth, target_lengths=None,
                      expansion_factor=expansion_factor, dropout=dropout,
                      normalize=True, residual=True, verbose=False),
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes),
    )

def ViShaker(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    shape = [image_size//patch_size, image_size//patch_size, patch_size, patch_size, channels]

    return nn.Sequential(
        Rearrange('b c (nph psh) (npw psw) -> b nph npw (psh psw c)', psh=patch_size, psw=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        Rearrange('b nph npw (psh psw c) -> b nph npw psh psw c', psh=patch_size, psw=patch_size),
        MLPFlatShaker(shape, [0, 1, 2, 3, 4]*depth, target_lengths=None,
                      expansion_factor=expansion_factor, dropout=dropout,
                      normalize=True, residual=True, verbose=False),
        nn.LayerNorm([patch_size, patch_size, channels]),
        Reduce('b n1 n2 p1 p2 c -> b (p1 p2 c)', 'mean'),
        nn.Linear(dim, num_classes),
    )

