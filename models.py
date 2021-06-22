
class MLPFlatShakerBlock(nn.Linear):
    """
    This module is simply a linear layer that is applied to an arbritrary dimension rather than the last dimension.
    """
    def __init__(self, in_features, out_features, bias=True, shape=None, dim_to_mix=None, verbose=False):
        super().__init__(in_features, out_features, bias)
        self.shape = shape
        self.dim_to_mix = dim_to_mix 
        
    def forward(self, x):
        # x has shape (b, *shape)
        n_dims_from_right = len(self.shape)-self.dim_to_mix-1
        x = x.reshape(-1, *self.shape[:self.dim_to_mix], 1, *self.shape[self.dim_to_mix:])
        
        weight = self.weight.reshape(*self.weight.shape, *[1]*(n_dims_from_right))
        x = (x*weight).sum(dim=-n_dims_from_right-1)
        if self.bias is not None:
            bias = self.bias.reshape(*self.bias.shape, *[1]*(n_dims_from_right))
            x = x+bias
        return x
    
class MLPFlatShaker(nn.Module):
    """
    Maps from (bs, N) to (bs, N)
    where N = flat_prod(dims).
    
    This reshapes N->*dims (for example 100->2,10,5), 
    then applies linear layers to different dims of this new shape.
    The dims_to_mix determines which dimensions are mixed in which order.
    """
    def __init__(self, shape, dims_to_mix, target_lengths=None, verbose=False):
        super().__init__()
        
        self.shape = shape
        self.dims_to_mix = dims_to_mix
        
        if target_lengths is None:
            target_lengths = [None]*len(self.dims_to_mix)
        self.target_lengths = target_lengths
        
        
        fshape = [length for length in self.shape]
        self.mix_dims = nn.ModuleList([])
        for dim, target_length in zip(self.dims_to_mix, self.target_lengths):
            if target_length is None:
                target_length = fshape[dim]
            m = MLPFlatShakerBlock(fshape[dim], target_length, shape=copy.copy(fshape), dim_to_mix=dim)
            fshape[dim] = target_length
            print(fshape)
            self.mix_dims.append(m)
        
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, *self.shape)
        for m in self.mix_dims:
            x = m(x)
        return x


class MLPShaker(nn.Module):
    """
    Maps from (bs, N) to (bs, N)
    where N = flat_prod(dims).
    """
    def __init__(self, dims, total_repeats, n_repeats=None, verbose=False):
        super().__init__()
        if n_repeats is None:
            n_repeats = total_repeats
        
        self.n_repeats = n_repeats
        self.dims = dims
        self.flat_dims = [flat_prod(d) for d in self.dims]
        
        total_depth = len(total_repeats)
        depth = len(self.n_repeats)
        s = '\t'*(total_depth-depth)
        print(f'{s}Decomposing (depth={depth}) {self.dims} -> {self.flat_dims}')
        print(f'{s}with repeats={self.n_repeats}')
        
        self.mixings = nn.ModuleList([])
        print(f'{s}----> {self.n_repeats[0]}x ')
        for i_repeat in range(self.n_repeats[0]):
            mixing = nn.ModuleList([])
            for d in self.dims:
                if type(d) is int:
                    if i_repeat==0:
                        print(f'{s}Putting Linear size {d}')
                    mixing.append(nn.Linear(d, d))
                else:
                    
                    if i_repeat==0:
                        mixing.append(MLPShakerBlock(d, total_repeats, self.n_repeats[1:]))
                    else:
                        with contextlib.redirect_stdout(None):
                            mixing.append(MLPShakerBlock(d, total_repeats, self.n_repeats[1:]))
                            
            self.mixings.append(mixing)
        print(f'{s}<----')
        
    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, *self.flat_dims)
        
        print(f'{shape} --> {x.shape}')
        for mixing in self.mixings:
            for dim, mix_dim in enumerate(mixing):
                dim+=1 # ignore batch size
                x = x.movedim(dim, -1)
#                 print(x.shape, 'dim=', dim)
                if type(mix_dim) is nn.Linear:
                    print(f'Densely mixing with {x.shape[-1]}')
                x = mix_dim(x)
                x = x.movedim(-1, dim)
        print(f'{shape} <-- {x.shape}')
        return x.reshape(shape)






