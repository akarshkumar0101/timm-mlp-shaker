import torch
from torch import nn

import copy

class DimLinear(nn.Linear):
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
    
class DimLayerNorm(nn.LayerNorm):
    """
    This module is simply a layer norm that is applied to an arbritrary dimension rather than the last dimension or all the dimensions.
    """
    def __init__(self, shape, dim_to_mix=None, elementwise_affine=True):
        self.input_shape = list(shape)
        self.normalized_shape = [1 for l in self.input_shape]
        self.normalized_shape[dim_to_mix] = self.input_shape[dim_to_mix]
        super().__init__(self.normalized_shape, elementwise_affine=elementwise_affine)
        
        self.dim_to_mix = dim_to_mix 
        
        self.n_dims_from_right = len(self.input_shape)-self.dim_to_mix-1
        
    def forward(self, x):
        assert list(x.shape[-len(self.input_shape):])==self.input_shape, 'input shape not correct'
        dim = -self.n_dims_from_right-1
        x = (x-x.mean(dim=dim, keepdim=True))/x.std(dim=dim, keepdim=True, unbiased=False)
        return x*self.weight+self.bias
    
    
    
