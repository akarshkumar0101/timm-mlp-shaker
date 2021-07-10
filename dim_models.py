import numpy as np
import torch
from torch import nn

import copy

import util

class DimLinearManual(nn.Linear):
    """
    This module is simply a linear layer that is applied to an arbritrary dimension rather than the last dimension.
    """
    def __init__(self, in_features, out_features=None, bias=True, shape=None, dim_to_mix=None, verbose=False):
        if out_features is None:
            out_features = in_features
        super().__init__(in_features, out_features, bias)
        
        self.input_shape = list(shape)
        self.output_shape = copy.copy(self.input_shape)
        assert self.input_shape[dim_to_mix]==in_features, 'input_shape at dim_to_mix does not match in_features'
        self.output_shape[dim_to_mix] = out_features
        self.dim_to_mix = dim_to_mix 
        
        self.n_dims_from_right = len(self.input_shape)-self.dim_to_mix-1
        self.weight.data= self.weight.reshape(*self.weight.shape, *[1]*(self.n_dims_from_right))
        if self.bias is not None:
            self.bias.data = self.bias.reshape(*self.bias.shape, *[1]*(self.n_dims_from_right))
        
    def forward(self, x):
        # x has shape (b, *shape)
        bs = x.shape[:len(x.shape)-len(self.input_shape)]
        
        # x: (99, 3, 2, 11, 2, 3)
        x = x.reshape(*bs, *self.input_shape[:self.dim_to_mix], 1, *self.input_shape[self.dim_to_mix:])
        # x: (99, 3, 2, 1, 11, 2, 3)
        
        x = (x*self.weight).sum(dim=-self.n_dims_from_right-1)
        # x: (99, 3, 2, 12, 2, 3)
        if self.bias is not None:
            x = x+self.bias
        # x: (99, 3, 2, 12, 2, 3)
        return x
    
class DimLinear(nn.Conv1d):
    """
    This module is simply a linear layer that is applied to an arbritrary dimension rather than the last dimension.
    """
    def __init__(self, shape, dim_to_mix, out_features=None, bias=True):
        self.in_features = shape[dim_to_mix]
        if out_features is None:
            out_features = self.in_features
        self.out_features = out_features
        super().__init__(self.in_features, self.out_features, kernel_size=1, bias=bias)
        
        self.input_shape = list(shape)
        self.output_shape = copy.copy(self.input_shape)
        self.output_shape[dim_to_mix] = out_features
        
        
        self.ndims = len(self.input_shape)
        self.dim_from_left = dim_to_mix if dim_to_mix>=0 else dim_to_mix+self.ndims
        self.dim_from_right = self.dim_from_left-self.ndims
        
    def forward(self, x, verbose=False):
        bs, is_ = util.bs_is_split(x.shape, np.prod(self.input_shape, dtype=int))
        assert list(is_) == self.input_shape, f'Input shape {is_} does not match expected input shape {self.input_shape}'
            
        bsis = [*bs, *self.input_shape]
        bsos = [*bs, *self.output_shape]
        left_collapse = np.prod([*bs, *is_[:self.dim_from_left]], dtype=int)
        right_collapse = np.prod([*is_[self.dim_from_left+1:]], dtype=int)
        proc_shape = [left_collapse, self.in_features, right_collapse]
        
        if verbose:
            print(f'collapsing shape {bsis}->{proc_shape}->{bsos}')
        
        x = x.reshape(*proc_shape)
        x = super().forward(x)
        x = x.reshape(*bsos)
        return x
    
class DimLayerNorm(nn.LayerNorm):
    """
    This module is simply a layer norm that is applied to arbritrary dimensions rather than the last dimension or all the dimensions.
    """
    def __init__(self, shape, dims_to_mix=[-1], eps=1e-5, elementwise_affine=True):
        if type(dims_to_mix) is int:
            dims_to_mix = [dims_to_mix]
            
        self.input_shape = list(shape)
        
        self.ndims = len(self.input_shape)
        self.dims_from_left = [(d if d>=0 else d+self.ndims) for d in dims_to_mix]
        self.dims_from_right = [d-self.ndims for d in self.dims_from_left]
        self.eps = eps
        
        self.normalized_shape = [(l if i in self.dims_from_left else 1) for i, l in enumerate(self.input_shape)]
        super().__init__(self.normalized_shape, elementwise_affine=elementwise_affine)
        
    def forward(self, x):
        assert list(x.shape[-len(self.input_shape):])==self.input_shape, 'input shape not correct'
        dims = tuple(self.dims_from_right)
        mean = x.mean(dim=dims, keepdim=True)
        std = (x.var(dim=dims, keepdim=True, unbiased=False)+self.eps).sqrt()
        x = (x-mean)/std
        return torch.addcmul(self.bias, x, self.weight)    
    
