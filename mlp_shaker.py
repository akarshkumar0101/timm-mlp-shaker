import torch
from torch import nn

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





