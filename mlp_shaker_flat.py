import numpy as np
import torch
from torch import nn

from einops.layers.torch import Rearrange, Reduce

import copy

import dim_models
import util

def FeedForward(in_features, out_features=None, bias=True, shape=None, dim_to_mix=None, 
                expansion_factor=2, dropout=0.):
    if out_features is None:
        out_features = in_features
    n_hidden = int(in_features*expansion_factor)
    shape1 = copy.copy(list(shape))
    shape2 = copy.copy(shape1)
    shape2[dim_to_mix] = n_hidden
    return nn.Sequential(
        dim_models.DimLinear(in_features, n_hidden, bias, shape1, dim_to_mix),
        nn.GELU(),
        nn.Dropout(dropout),
        dim_models.DimLinear(n_hidden, out_features, bias, shape2, dim_to_mix),
        nn.Dropout(dropout)
    )

class PreNormResidual(nn.Module):
    def __init__(self, shape, fn, dim_to_mix, normalize=True, residual=True):
        super().__init__()
        self.normalize = normalize
        self.residual = residual
        
        self.fn = fn
        self.norm = dim_models.DimLayerNorm(shape, dim_to_mix) if self.normalize else None

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
                 expansion_factor=1, dropout=0., normalize=True, residual=True, 
                 verbose=False):
        super().__init__()
        
        self.input_shape = shape
        self.dims_to_mix = dims_to_mix
        
        if target_lengths is None:
            target_lengths = [None]*len(self.dims_to_mix)
        self.target_lengths = target_lengths
        
        fshape = copy.copy(list(self.input_shape)) # forward shape (as its changing)
        self.mix_dims = nn.ModuleList([])
        for dim, target_length in zip(self.dims_to_mix, self.target_lengths):
            if target_length is None:
                target_length = fshape[dim]
#             m = DimLinear(fshape[dim], target_length, shape=copy.copy(fshape), dim_to_mix=dim)
            m = FeedForward(fshape[dim], target_length, shape=copy.copy(fshape), dim_to_mix=dim,
                            expansion_factor=expansion_factor, dropout=dropout)
            m = PreNormResidual(fshape, m, dim_to_mix=dim, normalize=normalize, residual=residual)
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
    
class ViShaker(nn.Module):
    """
    TODO add doc
    """
    def __init__(self, shape, dims_to_mix, target_lengths=None, 
                 expansion_factor=1, dropout=0., normalize=True, residual=True,
                 global_avg_pool_str=None, num_classes=10,
                 verbose=False):
        super().__init__()
        
        self.to_reshape = Rearrange('b c (nph psh) (npw psw) -> b (nph npw) (psh psw c)', nph=8, npw=8)
        
        self.shaker = MLPFlatShaker(shape, dims_to_mix, target_lengths=target_lengths, 
                                    expansion_factor=expansion_factor, dropout=dropout,
                                    normalize=normalize, residual=residual,
                                    verbose=verbose)
        
        self.global_avg_pool = Reduce(global_avg_pool_str, 'mean')
        
        self.classification_head = nn.Linear(48, num_classes)
        
    def forward(self, x):
#         x = self.to_patch_embedding(x)
        x = self.to_reshape(x)
        x = self.shaker(x)
        x = self.global_avg_pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.classification_head(x)
        return x.log_softmax(dim=-1)
