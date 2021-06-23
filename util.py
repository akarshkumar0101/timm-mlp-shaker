import numpy as np

def flatten_list(a):
    ret = []
    if type(a) is list:
        for i in a:
            ret.extend(flatten_list(i))
    else:
        ret.append(a)
    return ret
def flat_prod(a):
    return np.prod(flatten_list(a), dtype=int)

def count_params(net):
    return np.sum([p.numel() for p in net.parameters()], dtype=int)