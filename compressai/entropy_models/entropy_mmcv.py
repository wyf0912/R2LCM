import torch
from mmcv.ops import rans_encode_with_indexes, rans_decode_with_indexes, pmf_to_quantized_cdf
import time

MIN_V = -20
MAX_V = 20
TAIL_MASS = 1e-6
NUM_THREADS = 10
N = 100000

if __name__ == '__main__':
    t_s = time.time_ns()
    # define a GMM
    mean, std, k = [torch.randn([N, 3]) for _ in range(3)]
    std = torch.abs(std) + 1e-6
    k = torch.sigmoid(k)
    # define cdf range    
    v = torch.arange(MIN_V, MAX_V + 1)[None, None, :] # used for define the range of the symbol
    tail_mass = torch.ones([N]) * TAIL_MASS # used for define the tail mass of the symbol
    # define a symbol
    symbol = torch.randint(-20, 20, [N]).int() # the number of symbol should be the same as the gmms

    # generate the cdf & indexes & offsets
    m = torch.distributions.normal.Normal(mean[:, :, None], std[:, :, None])
    lower = m.cdf(v - 0.5) # [N, 3, MAX_V - MIN_V + 1]
    upper = m.cdf(v + 0.5) # [N, 3, MAX_V - MIN_V + 1]
    pmf = ((upper - lower) * k[:, :, None]).sum(1) # [N, MAX_V - MIN_V + 1]
    pmf_length = torch.ones(N).int() * (MAX_V - MIN_V + 1) # [N]
    # the cdf is a two-dimensional array, the first dimension is the number of gmms, the second dimension is MAX_V - MIN_V + 1 + 2
    # so the first value of the cdf is zero, the last value of the cdf is "one" (quantized into 65536)
    # the probability between the cdfs[:, 0] cdfs[:, 1] is the MIN_V, the probability between the cdfs[:, -2] cdfs[:, -1] is the MAX_V 
    cdfs = pmf_to_quantized_cdf(pmf, pmf_length, tail_mass) # [100, MAX_V - MIN_V + 1 + 2]
    cdfs_sizes = pmf_length + 2 # [N]
    indexes = torch.arange(N).int() # [N] # one-to-one correspondence with the cdfs
    offsets = torch.ones(N).int() * MIN_V # [N]

    torch.cuda.synchronize()
    print((time.time_ns()-t_s)/(10**9))
    t_s = time.time_ns()

    # encode
    bitstreams =rans_encode_with_indexes(symbol, indexes, cdfs, cdfs_sizes, offsets, NUM_THREADS)
    # decode
    decoded_symbols = rans_decode_with_indexes(bitstreams, indexes, cdfs, cdfs_sizes, offsets)
    
    assert torch.equal(symbol, decoded_symbols)
    print((time.time_ns()-t_s)/(10**9))