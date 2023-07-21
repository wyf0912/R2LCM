# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warnings

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch import Tensor
import torch.multiprocessing as mp
from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound
try:
    import mmcv.ops
except:
    Warning("MMCV not installed.")

class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any) -> Any:
    raise NotImplementedError()

MIN_V = -20
MAX_V = 20
TAIL_MASS = 1e-6
NUM_THREADS = 1

class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()
        self.num_processes = 20
        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None, quantization_level_map: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols", "forward", "gumbel"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')
        outputs = inputs.clone()

        mutiplier = 1
        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            if quantization_level_map is not None:
                mutiplier = torch.exp2(quantization_level_map)
            outputs = inputs + noise * mutiplier
            return outputs
        else:
            if means is not None:
                outputs = outputs - means

            if quantization_level_map is not None:
                outputs = outputs * torch.exp2(-quantization_level_map)
                mutiplier = torch.exp2(quantization_level_map)
                
            if mode == "forward":
                outputs = ((outputs.round() - outputs).detach() + outputs) * mutiplier + means
            elif mode == "gumbel":
                outputs_lower = outputs.floor()
                outputs_upper = outputs.ceil()
                mask = torch.rand_like(outputs) < (outputs - outputs_lower)  
                target = (~mask) * outputs_lower + mask*outputs_upper
                outputs = ((target - outputs).detach() + outputs)*mutiplier + means
            elif mode == "dequantize":
                outputs = torch.round(outputs) * mutiplier
                if means is not None:
                    outputs += means
            elif mode == "symbols":
                outputs = torch.round(outputs)
                outputs = outputs.int()
        
            return outputs

    def _quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        warnings.warn("_quantize is deprecated. Use quantize instead.")
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Optional[Tensor] = None, dtype: torch.dtype = torch.float, quantization_level_map: Optional[Tensor] = None
    ) -> Tensor:
        if quantization_level_map is not None:
            inputs = inputs * torch.exp2(quantization_level_map)
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor] = None) -> Tensor:
        warnings.warn("_dequantize. Use dequantize instead.")
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")
    
    def compress_gmm(self, inputs, means, scales, probs, quantization_level_map=None):
        # inputs shape B*C*N 
        strings = []
        for i in range(inputs.shape[0]): # batch
            symbols = self.quantize(inputs[i], "symbols", quantization_level_map=quantization_level_map[i])
            mutipler = torch.exp2(-quantization_level_map[i])
            scales_ = (scales[:,i]*mutipler).clip(self.lower_bound_scale.bound.item())
            means_ = (means[:,i]*mutipler)
            v = torch.arange(MIN_V, MAX_V + 1)[None,None,None,:].to(inputs.device)
            m = torch.distributions.normal.Normal(means_.unsqueeze(-1), scales_.unsqueeze(-1))
            lower = m.cdf(v - 0.5) # [N, 3, MAX_V - MIN_V + 1]
            upper = m.cdf(v + 0.5) # [N, 3, MAX_V - MIN_V + 1]
            pmf = ((upper - lower) * probs[:, i].unsqueeze(-1)).sum(dim=0)
            N = pmf.shape[0]*pmf.shape[1]
            pmf_length = (torch.ones(pmf.shape[0]*pmf.shape[1]).int() * (MAX_V - MIN_V + 1)).to(inputs.device)
            tail_mass = torch.ones_like(pmf_length) * self.tail_mass
            cdfs = mmcv.ops.pmf_to_quantized_cdf(pmf.view(-1,41), pmf_length, tail_mass)
            cdfs_sizes = pmf_length + 2 # [N]
            indexes = torch.arange(N).int().to(inputs.device) # [N] # one-to-one correspondence with the cdfs
            offsets = torch.ones(N).int().to(inputs.device) * MIN_V # [N]
            # encode
            bitstreams = mmcv.ops.rans_encode_with_indexes(symbols.view(-1), indexes, cdfs, cdfs_sizes, offsets, NUM_THREADS)
            strings.append(bitstreams)
        return strings
    
    def decompress_gmm(self, strings, means, scales, probs, quantization_level_map=None):
        # inputs shape B*C*N 
        outputs = torch.zeros_like(quantization_level_map)
        for i in range(means.shape[1]): # batch
            if len(strings[i]) != 0:
                mutipler = torch.exp2(-quantization_level_map[i])
                scales_ = (scales[:,i]*mutipler).clip(self.lower_bound_scale.bound.item())
                means_ = (means[:,i]*mutipler)
                v = torch.arange(MIN_V, MAX_V + 1)[None,None,None,:].to(means.device)
                m = torch.distributions.normal.Normal(means_.unsqueeze(-1), scales_.unsqueeze(-1))
                lower = m.cdf(v - 0.5) # [N, 3, MAX_V - MIN_V + 1]
                upper = m.cdf(v + 0.5) # [N, 3, MAX_V - MIN_V + 1]
                pmf = ((upper - lower) * probs[:, i].unsqueeze(-1)).sum(dim=0)
                N = pmf.shape[0]*pmf.shape[1]
                pmf_length = (torch.ones(pmf.shape[0]*pmf.shape[1]).int() * (MAX_V - MIN_V + 1)).to(means.device)
                tail_mass = torch.ones_like(pmf_length) * self.tail_mass
                cdfs = mmcv.ops.pmf_to_quantized_cdf(pmf.view(-1,41), pmf_length, tail_mass)
                cdfs_sizes = pmf_length + 2 # [N]
                indexes = torch.arange(N).int().to(means.device) # [N] # one-to-one correspondence with the cdfs
                offsets = torch.ones(N).int().to(means.device) * MIN_V # [N]
                # encode
                results = mmcv.ops.rans_decode_with_indexes(strings[i], indexes, cdfs, cdfs_sizes, offsets)
                outputs[i] = results.view(outputs.shape[1:])
        outputs = outputs * torch.exp2(quantization_level_map)
        return outputs
    
    def compress(self, inputs, indexes, means=None, multiprocess=False, quantization_level_map=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means, quantization_level_map=quantization_level_map).clip(-32768, 32767)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):
            if multiprocess:
                manager = mp.Manager()
                q = manager.Queue()
                processes = []
                def compres_subprocess(queue, rank, symbols, index,  *args, **kwargs):
                    symbols = symbols[split[rank]:split[rank+1]].tolist()
                    index = index[split[rank]:split[rank+1]].tolist()
                    string = self.entropy_coder.encode_with_indexes(symbols, index, *args, **kwargs)
                    queue.put([rank, string])
                
                l_symbols = symbols[i].reshape(-1).int().cpu() # .tolist()    
                l_indexes = indexes[i].reshape(-1).int().cpu() # .tolist()
                l_qcdf = self._quantized_cdf.tolist()
                l_cdf_l = self._cdf_length.reshape(-1).int().tolist()
                l_offset = self._offset.reshape(-1).int().tolist()
                # l_symbols.share_memory_()
                # l_indexes.share_memory_()
                p_num = len(symbols[i].reshape(-1))
                split = np.linspace(0, p_num, self.num_processes+1).astype(int)
                for rank in range(self.num_processes):
                    p = mp.Process(target=compres_subprocess, args=(q, rank, 
                        l_symbols,
                        l_indexes,
                        l_qcdf,
                        l_cdf_l,
                        l_offset,))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                res = [q.get() for i in range(self.num_processes)]
                rv = sorted(res, key=lambda x: x[0])
            else:
                rv = self.entropy_coder.encode_with_indexes(
                    symbols[i].reshape(-1).int().tolist(),
                    indexes[i].reshape(-1).int().tolist(),
                    self._quantized_cdf.tolist(),
                    self._cdf_length.reshape(-1).int().tolist(),
                    self._offset.reshape(-1).int().tolist(),
                )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
        multiprocess = False,
        quantization_level_map=None
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            if multiprocess:
                manager = mp.Manager()
                q = manager.Queue()
                processes = []
                def decompress_subprocess(queue, rank, *args, **kwargs):
                    string = self.entropy_coder.decode_with_indexes(*args, **kwargs)
                    queue.put([rank, string])
                  
                l_indexes = indexes[i].reshape(-1).int().tolist()
                
                p_num = len(l_indexes)
                split = np.linspace(0, p_num, self.num_processes+1).astype(int)
                for rank in range(self.num_processes):
                    p = mp.Process(target=decompress_subprocess, args=(q, rank, s[rank][1],
                        l_indexes[split[rank]:split[rank+1]],
                        cdf.tolist(),
                        self._cdf_length.reshape(-1).int().tolist(),
                        self._offset.reshape(-1).int().tolist()))
                    p.start()
                    processes.append(p)
                for p in processes:
                    p.join()
                res = [q.get() for i in range(self.num_processes)]
                rv = sorted(res, key=lambda x: x[0])
                outputs[i] = torch.tensor(list(itertools.chain(*[v[1] for v in rv])), device=outputs.device, dtype=outputs.dtype).reshape(outputs[i].size())
            else:
                values = self.entropy_coder.decode_with_indexes(
                    s,
                    indexes[i].reshape(-1).int().tolist(),
                    cdf.tolist(),
                    self._cdf_length.reshape(-1).int().tolist(),
                    self._offset.reshape(-1).int().tolist(),
                )
                outputs[i] = torch.tensor(
                    values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype, quantization_level_map=quantization_level_map)
        return outputs


# import torchac
# import torch
# import torch.nn as nn
# import math

# class FastArithmetic:
#     def __init__(self, maxval=5):
#         """_summary_

#         Args:
#             maxval (int, optional): . Defaults to 5.
#         """
#         self.maxval = maxval
#         self.symbols = torch.linspace(-self.maxval, self.maxval, steps=self.maxval*2+1).cuda()
#         pass
    
#     def compress(self, x, means, std):
#         symbols_input = torch.round(x - means).clip(-self.maxval, self.maxval)+self.maxval-1
#         anchor = self.symbols[(None, )*len(means.shape)]
#         cdf = 0.5*(1+torch.erf((anchor+0.5)/(std.unsqueeze(-1)*math.sqrt(2))))
#         cdf[...,0] = 0
#         cdf[...,-1] = 1
#         return [torchac.encode_float_cdf(cdf.cpu(), symbols_input.type(torch.int16).cpu(),check_input_bounds=True)]
        
#     def decompress(self, bitstream, means, std):
#         anchor = self.symbols[(None, )*len(means.shape)]
#         cdf = 0.5*(1+torch.erf((anchor+0.5)/(std.unsqueeze(-1)*math.sqrt(2))))
#         cdf[...,0] = 0
#         cdf[...,-1] = 1
#         sym_out = torchac.decode_float_cdf(cdf.cpu(), bitstream[0])-self.maxval+1
#         return (sym_out.cuda()+means)

class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        rounding: str = "noise",
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)
        self.filters = tuple(int(f) for f in filters)
        self.init_scale = float(init_scale)
        self.tail_mass = float(tail_mass)
        self.rounding = rounding
        # Create parameters
        filters = (1,) + self.filters + (1,)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))
        channels = self.channels

        for i in range(len(self.filters) + 1):
            init = np.log(np.expm1(1 / scale / filters[i + 1]))
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1)

        target = np.log(2 / self.tail_mass - 1)
        self.register_buffer("target", torch.Tensor([-target, 0, target]))

    def _get_medians(self): #  -> Tensor:
        medians = self.quantiles[:, :, 1:2]
        return medians

    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:
            return False

        medians = self.quantiles[:, 0, 1]

        minima = medians - self.quantiles[:, 0, 0]
        minima = torch.ceil(minima).int()
        minima = torch.clamp(minima, min=0)

        maxima = self.quantiles[:, 0, 2] - medians
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima

        pmf_start = medians - minima
        pmf_length = maxima + minima + 1

        max_length = pmf_length.max().item()
        device = pmf_start.device
        samples = torch.arange(max_length, device=device)

        samples = samples[None, :] + pmf_start[:, None, None]

        half = float(0.5)

        lower = self._logits_cumulative(samples - half, stop_gradient=True)
        upper = self._logits_cumulative(samples + half, stop_gradient=True)
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))

        pmf = pmf[:, 0, :]
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])

        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._cdf_length = pmf_length + 2
        return True

    def loss(self) -> Tensor:
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)
        loss = torch.abs(logits - self.target).sum()
        return loss

    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        for i in range(len(self.filters) + 1):
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self.filters):
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)
        return logits

    @torch.jit.unused
    def _likelihood(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)
        sign[sign==0] = 1
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)
        )
        return likelihood

    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))
            perm[0], perm[1] = perm[1], perm[0]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()
        values = x.reshape(x.size(0), 1, -1)

        # Add noise or quantize
        if self.rounding == "gumbel":
            outputs = self.quantize(values, "gumbel", self._get_medians())
        elif self.rounding == "noise":
            outputs = self.quantize(values, "noise" if training else "dequantize", self._get_medians())
        elif self.rounding == "forward":
            outputs = self.quantize(values, "forward" if training else "dequantize", self._get_medians())
        else:
            raise NotImplementedError(f"The rounding policy '{self.rounding}' is not supported")
        # outputs = self.quantize(
        #     values, "noise" if training else "dequantize", self._get_medians()
        # )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):
        dims = len(size)
        N = size[0]
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x, *args, **kargs):
        indexes = self._build_indexes(x.size())
        medians = self._get_medians().detach()
        spatial_dims = len(x.size()) - 2
        medians = self._extend_ndims(medians, spatial_dims)
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))
        return super().compress(x, indexes, medians, *args, **kargs)

    def decompress(self, strings, size, *args, **kargs):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians, *args, **kargs)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and (
            scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)
        ):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)
        # print(f"[INFO] Using the rounding policy '{self.rounding}'")
        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        const = float(-(2**-0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs) # entropy

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)

    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)
        self.update()
        return True

    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2) # CDF的逆函数 # tail mass:高斯分布两侧的尾巴
        pmf_center = torch.ceil(self.scale_table * multiplier).int() # scale_table: 不同的标准差
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None] 
        ) # 计算没有归一化前 距离不同center的距离 多余的部分(normalize以后超过multiplie的)会在后面被裁掉
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale) # 把分位的点变回到标准高斯分布
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1] # 准确的tail mass

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None, quantization_level_map: Optional[Tensor] = None, prob: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)
        
        scales = self.lower_bound_scale(scales)

        
        if quantization_level_map is not None:
            mutiplier = torch.exp2(quantization_level_map)
        else:
            mutiplier = 1
        if len(means.shape) == 4:
            if means is not None:
                values = inputs - means
            else:
                values = inputs
            values = torch.abs(values)
            
            upper = self._standardized_cumulative((mutiplier*half - values) / scales)
            lower = self._standardized_cumulative((-mutiplier*half - values) / scales)
            likelihood = upper - lower
        elif len(means.shape) == 5: # GMM
            gmm_num = means.shape[0]
            likelihood = torch.zeros_like(inputs)
            for i in range(gmm_num):
                upper = self._standardized_cumulative((mutiplier*half - (inputs - means[i])) / scales[i])
                lower = self._standardized_cumulative((-mutiplier*half - (inputs - means[i])) / scales[i])
                likelihood += ((upper - lower) * prob[i])
        else:
            raise NotImplemented(f"The shape of {means.shape} is not supported.")
        return likelihood

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        quantization_level_map = None,
        training: Optional[bool] = None,
        prob: Optional[Tensor] = None,
        soft: Optional[bool] = False,
        rounding: Optional[str] = "noise"
    ) -> Tuple[Tensor, Tensor]:
        if len(means.shape) == 5:
            means_for_quant = 0 # For GMM
        else:
            means_for_quant = means
            
        if training is None:
            training = self.training
        if rounding == "gumbel":
            outputs = self.quantize(inputs, "gumbel", means_for_quant, quantization_level_map=quantization_level_map)
        elif rounding == "noise" or soft == True:
            outputs = self.quantize(inputs, "noise" if training else "dequantize", means_for_quant, quantization_level_map=quantization_level_map)
        elif rounding == "forward":
            outputs = self.quantize(inputs, "forward" if training else "dequantize", means_for_quant, quantization_level_map=quantization_level_map)
        elif rounding == "none":
            outputs = inputs
        else:
            raise NotImplementedError(f"The rounding policy '{self.rounding}' is not supported")
        likelihood = self._likelihood(outputs, scales, means, quantization_level_map, prob=prob)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
            
        # if self.rounding == "forward" and training:
        #     outputs = self.quantize(inputs, "noise", quantization_level_map=quantization_level_map)
            
        return outputs, likelihood

    def build_indexes(self, scales: Tensor, quantization_level_map: Optional[Tensor] = None) -> Tensor:
        if quantization_level_map is not None:
            scales = torch.exp2(-quantization_level_map) * scales
        scales = self.lower_bound_scale(scales)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes
