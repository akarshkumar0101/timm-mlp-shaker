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

def bs_is_split(shape, prod):
    """
    Get the batch_shape and the input_shape from the shape given that the input_shape product is prod.
    """
    prods = [np.prod(shape[::-1][:n], dtype=int) for n in range(1, len(shape)+1)]
    match = np.array(prods)==prod
    if match.sum()==0:
        raise Exception(f'This shape: {shape} is invalid for the proposed input shape product {prod}')
    n = np.argmax(match)+1
    return shape[:-n], shape[-n:]
