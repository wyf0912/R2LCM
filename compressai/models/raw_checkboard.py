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

from enum import auto
from locale import normalize
from torchvision.models.resnet import ResNet
from mimetypes import init
from tkinter import Y
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
from torchvision.utils import save_image
from compressai.models.google import JointAutoregressiveHierarchicalPriors, CompressionModel, get_scale_table
from compressai.models.waseda import Cheng2020Attention
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d
import warnings
from compressai.models.utils import conv, deconv, update_registered_buffers
from typing import Any
from torch import Tensor
from compressai.layers import MaskedConv2d, CheckerboardContext
from tqdm import tqdm
import torch
import time

def auto_padding(img, times=32):
    if len(img.shape)==3:
        img = img.unsqueeze(0)
    # img: torch tensor image with shape H*W*C
    b, c, h, w = img.shape
    h1, w1 = (times - h % times) // 2, (times - w % times) // 2
    h2, w2 = (times - h % times) - h1, (times - w % times) - w1
    img = F.pad(img, [w1, w2, h1, h2], "reflect")
    return img, [w1, w2, h1, h2]


class MaskedDeconv(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        kernel_size = kwargs.pop("kernel_size")
        padding = kernel_size//2
        self.kernel_size=kernel_size
        self.deconv = nn.ConvTranspose2d(kernel_size=kernel_size, padding=padding, stride=1, *args, **kwargs)
        # self.count = nn.Conv2d(kernel_size=kernel_size, padding=padding, in_channels=1, out_channels=1,stride=1)
        
    def forward(self, input):
        input, mask = input[:,:-1], input[:,[-1]]
        deconved = self.deconv(input)
        assert deconved.shape[2:]==input.shape[2:]
        with torch.no_grad():
            count_mask = F.conv2d(mask, weight=torch.ones(1,1,self.kernel_size, self.kernel_size).to(input.device), padding=self.kernel_size//2)
        deconved = deconved/count_mask.clip(min=1)
        return deconved
    
class JpegConditionedSequential(nn.Sequential):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*kargs, **kwargs)

    def forward(self, input, x_jpg):
        for idx, module in enumerate(self):
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, AttentionBlock, MaskedDeconv)):
                input = module(input)
            else:
                input = module(
                    torch.cat([F.interpolate(x_jpg, size=input.shape[2:]), input], dim=1))
        return input



        
class RawCheckboardContext(Cheng2020Attention):
    def __init__(self, N=192, **kwargs):
        for useless_arg in ["noquant", "hyper_mean", "resid", "relu", "xyz"]:
            kwargs.pop(useless_arg, False)
            
        self.ablation_no_raw = kwargs.pop("ablation_no_raw", False)
        self.demosaic = kwargs.pop("demosaic", False)
        self.stride = kwargs.pop("stride", 2)
        self.reduce_c = reduce_c = kwargs.pop("reduce_c", 1)
        self.resid_path = kwargs.pop("resid_path", False)
        self.context_z = kwargs.pop("context_z", False)
        self.discrete_context = kwargs.pop("discrete_context", False)
        self.down_num = kwargs.pop("down_num", 4)
        self.channel_mean = kwargs.pop("channel_mean", False)
        self.sampling_num = kwargs.pop("sampling_num", 2) # The number of sampling for context model
        self.use_deconv = kwargs.pop("use_deconv", False)   
         
        self.embed_weight = True
        super().__init__(N=N, **kwargs)
        
        stride_num = 4
        stride_list = [self.stride if i < self.down_num else 1 for i in range(stride_num)]
        self.raw_channel = 3 if self.demosaic else 4
        self.gaussian_conditional = GaussianConditional(None,likelihood_bound=1e-6, tail_mass=1e-9, scale_bound=0.11)
        
        self.g_a = JpegConditionedSequential(
            ResidualBlockWithStride(self.raw_channel+3, N, stride=stride_list[0]),
            ResidualBlock(N+3, N),
            ResidualBlockWithStride(N+3, N, stride=stride_list[1]),
            AttentionBlock(N),
            ResidualBlock(N+3, N),
            ResidualBlockWithStride(N+3, N, stride=stride_list[2]),
            ResidualBlock(N+3, N),
            conv3x3(N+3, N// reduce_c, stride=stride_list[3]),
            AttentionBlock(N// reduce_c),
        )


        self.g_s = JpegConditionedSequential(
            AttentionBlock(N//reduce_c + (0 if not self.resid_path else self.raw_channel + 1)),
            ResidualBlock(N//reduce_c + (3 if not self.resid_path else self.raw_channel + 4), N),
            ResidualBlockUpsample(N+3, N, stride_list[3]),
            ResidualBlock(N+3, N),
            ResidualBlockUpsample(N+3, N, stride_list[2]),
            AttentionBlock(N),
            ResidualBlock(N+3, N),
            ResidualBlockUpsample(N+3, N, stride_list[1]),
            ResidualBlock(N+3, N),
            subpel_conv3x3(N+3, 4 if not self.demosaic else 3, stride_list[0]),
        )
        if self.resid_path:
            self.resid_path_net = nn.Sequential(
                nn.Conv2d(3, N * 10 // 3, 1),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(N * 8 // 3, 2, 1),
            )
            
        self.h_a = JpegConditionedSequential(
            conv3x3(N// reduce_c+3, N),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N, stride=2),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N// reduce_c, stride=2),
        )

        self.h_s = JpegConditionedSequential(
            conv3x3(N// reduce_c+3, N),
            nn.LeakyReLU(inplace=False),
            subpel_conv3x3(N+3, N, 2),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N * 3 // 2),
            nn.LeakyReLU(inplace=False),
            subpel_conv3x3(N * 3 // 2 + 3, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=False),
            conv3x3(N * 3 // 2 + 3, N // reduce_c * 2),
        )
        self.h_mask = JpegConditionedSequential(
            conv3x3(N // reduce_c * 2+3, N),
            nn.LeakyReLU(inplace=False),
            conv3x3(N+3, N * 3 // 2),
            nn.LeakyReLU(inplace=False),
            conv3x3(N * 3 // 2 + 3, self.sampling_num),
        )
        
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(12, N * 10 // 3, 1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(N * 8 // 3, N * 6 // 3 // reduce_c, 1),
        )
        
        self.context_prediction = CheckerboardContext(
            in_channels = N // reduce_c, out_channels = N // reduce_c, kernel_size=5, stride=1, padding=2
        )
    

        if self.context_z:
            self.gaussian_conditional_z = GaussianConditional(None)
            self.entropy_parameters_z = nn.Sequential(
                nn.Conv2d(N * 12 // 3 // reduce_c, N * 10 // 3, 1),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(N * 10 // 3, N * 8 // 3, 1),
                nn.LeakyReLU(inplace=False),
                nn.Conv2d(N * 8 // 3, N * 2 // reduce_c, 1),
            )
            self.context_prediction_z = MaskedConv2d(
                N// reduce_c, N// reduce_c, kernel_size=5, padding=2, stride=1
            )
            self.h_z = nn.Sequential(
                conv3x3(3, N),
                nn.LeakyReLU(inplace=False),
                conv3x3(N, N),
                nn.LeakyReLU(inplace=False),
                conv3x3(N, N),
                nn.LeakyReLU(inplace=False),
                conv3x3(N, N // reduce_c * 2),
            )
        else:
            self.entropy_bottleneck = EntropyBottleneck(N// reduce_c)
            
    def load_state_dict(self, state_dict, **kwargs):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        # if self.contex_z:
        #     update_registered_buffers(
        #         self.gaussian_conditional_z,
        #         "gaussian_conditional_z",
        #         ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
        #         state_dict,
        #     )
        return super().load_state_dict(state_dict, **kwargs)
        
    def forward(self, x_raw, x_jpg):
        if self.ablation_no_raw:
            x_raw = x_raw * 0
            y = self.g_a(x_raw, x_jpg)
            x_hat = self.g_s(y, x_jpg)
            return {"x_hat": x_hat, "likelihoods":{"y":torch.ones(1,1,1,1).cuda(), "z":torch.ones(1,1,1,1).cuda()}}
        # y = self.g_a(x_jpg/(x_raw+1e-3) if self.embed_weight else x_raw, x_jpg)
        y = self.g_a(x_raw, x_jpg)
        z = self.h_a(y, x_jpg)
        if self.context_z:
            params_z = self.h_z(F.interpolate(x_jpg,z.shape[2:]))
            z_hat = self.gaussian_conditional_z.quantize(
                z, "noise" if self.training else "dequantize"
            )
            ctx_params_z = self.context_prediction_z(z_hat)
            gaussian_params_z = self.entropy_parameters_z(
                torch.cat((params_z, ctx_params_z), dim=1)
            )
            scales_hat_z, means_hat_z = gaussian_params_z.chunk(2, 1)
            _, z_likelihoods = self.gaussian_conditional_z(z, scales_hat_z, means=means_hat_z)
        else:
            if self.channel_mean:
                z_mean = z.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
                z = z - z_mean
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
                z_hat = z_hat + z_mean
            else:
                z_hat, z_likelihoods = self.entropy_bottleneck(z)
        
        params = self.h_s(z_hat, x_jpg)
        
        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)
        ctx_params[:, :, 0::2, 1::2] = 0
        ctx_params[:, :, 1::2, 0::2] = 0
        
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, params], dim=1))
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat, x_jpg)
        
        res_dict = {}
        likelihoods = {"y":y_likelihoods, "z":z_likelihoods}
        res_dict.update({
            "x_hat": x_hat,
            "x_hat2": None,
            "likelihoods": likelihoods,
            "latent": {"y_hat": y_hat, "z_hat": z_hat},
            "gaussian_params": gaussian_params,
        })
        return res_dict

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        if self.context_z:
            updated_z = self.gaussian_conditional_z.update_scale_table(scale_table, force=force)
            updated = super().update(force=force) | updated_z
        return updated
 
    def compress(self, x_raw, x_jpg, **kwargs):
        torch.backends.cudnn.deterministic = True
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        
        y = self.g_a(x_raw, x_jpg)
        z = self.h_a(y, x_jpg)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2
        

        if not self.context_z:
            z_mean = z.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True) if self.channel_mean else 0
            z_strings = self.entropy_bottleneck.compress(z-z_mean)
            z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:]) + z_mean
        else:
            params_z = self.h_z(F.interpolate(x_jpg, z.shape[2:]))
            z_hat = F.pad(z, (padding, padding, padding, padding))
            # 
            z_strings = []
            z_height = z.size(2)
            z_width = z.size(3)
            for i in range(y.size(0)):
                string = self._compress_ar(
                    z_hat[i : i + 1],
                    params_z[i : i + 1],
                    z_height,
                    z_width,
                    kernel_size,
                    padding,
                    self.gaussian_conditional_z,
                    self.entropy_parameters_z,
                    self.context_prediction_z
                )
                z_strings.append(string)
            shape = z.size()[-2:]
            z_height = shape[0]
            z_width = shape[1]

            z_hat = F.pad(z_hat, (-padding, -padding, -padding, -padding))


        hyper_params = self.h_s(z_hat, x_jpg)

        # y_height = z_hat.size(2) * s
        # y_width = z_hat.size(3) * s

        # y_hat = F.pad(y, (padding, padding, padding, padding))
        y_strings = []
    
        hyper_params = self.h_s(z_hat, x_jpg)
        ctx_params_anchor = torch.zeros([y.size(0), 4, y.size(2), y.size(3)], device=y.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.compress_anchor(y, scales_anchor, means_anchor, symbols_list, indexes_list)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.compress_nonanchor(y, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)
        
        res =  {"strings": [[y_strings], z_strings], "shape": z.size()[-2:], "z_mean": z_mean}
        return res
    
    def decompress(self, strings, shape, x_jpg, z_mean=None, compressed_result=None):
        assert isinstance(strings, list) # and len(strings) == 2

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder
        torch.backends.cudnn.deterministic = True

        torch.cuda.synchronize()
        start_time = time.process_time()
        y_strings = strings[0][0][0]
        z_strings = strings[1]

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)
        
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        if not self.context_z:
            z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
            if self.channel_mean:
                z_hat += z_mean
        else:
            z_height = shape[0]
            z_width = shape[1]
            z_hat = torch.zeros(
                (len(strings[1]), self.M//self.reduce_c, z_height + 2 * padding, z_width + 2 * padding),
                device=x_jpg.device,
            )
            params_z = self.h_z(F.interpolate(x_jpg, z_hat.shape[2:]))
            for i, z_string in enumerate(strings[1]):
                self._decompress_ar(
                    z_string,
                    z_hat[i : i + 1],
                    params_z[i : i + 1],
                    z_height,
                    z_width,
                    kernel_size,
                    padding, self.gaussian_conditional_z, self.context_prediction_z, self.entropy_parameters_z
                )
            z_hat = F.pad(z_hat, (-padding, -padding, -padding, -padding))
            
        hyper_params = self.h_s(z_hat, x_jpg)
        ctx_params_anchor = torch.zeros([z_hat.size(0), 4, z_hat.size(2) * 4, z_hat.size(3) * 4], device=z_hat.device)
        gaussian_params_anchor = self.entropy_parameters(torch.cat([ctx_params_anchor, hyper_params], dim=1))
        scales_anchor, means_anchor = gaussian_params_anchor.chunk(2, 1)
        anchor_hat = self.decompress_anchor(scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)

        ctx_params = self.context_prediction(anchor_hat)
        gaussian_params = self.entropy_parameters(torch.cat([ctx_params, hyper_params], dim=1))
        scales_nonanchor, means_nonanchor = gaussian_params.chunk(2, 1)
        nonanchor_hat = self.decompress_nonanchor(scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)

        y_hat = anchor_hat + nonanchor_hat
    
        x_hat = self.g_s(y_hat, x_jpg).clamp_(0, 1)
        return {"x_hat": x_hat}
    
    
    
    def ckbd_anchor_sequeeze(self, y):
        B, C, H, W = y.shape
        anchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        anchor[:, :, 0::2, :] = y[:, :, 0::2, 1::2]
        anchor[:, :, 1::2, :] = y[:, :, 1::2, 0::2]
        return anchor

    def ckbd_nonanchor_sequeeze(self, y):
        B, C, H, W = y.shape
        nonanchor = torch.zeros([B, C, H, W // 2]).to(y.device)
        nonanchor[:, :, 0::2, :] = y[:, :, 0::2, 0::2]
        nonanchor[:, :, 1::2, :] = y[:, :, 1::2, 1::2]
        return nonanchor

    def ckbd_anchor_unsequeeze(self, anchor):
        B, C, H, W = anchor.shape
        y_anchor = torch.zeros([B, C, H, W * 2]).to(anchor.device)
        y_anchor[:, :, 0::2, 1::2] = anchor[:, :, 0::2, :]
        y_anchor[:, :, 1::2, 0::2] = anchor[:, :, 1::2, :]
        return y_anchor

    def ckbd_nonanchor_unsequeeze(self, nonanchor):
        B, C, H, W = nonanchor.shape
        y_nonanchor = torch.zeros([B, C, H, W * 2]).to(nonanchor.device)
        y_nonanchor[:, :, 0::2, 0::2] = nonanchor[:, :, 0::2, :]
        y_nonanchor[:, :, 1::2, 1::2] = nonanchor[:, :, 1::2, :]
        return y_nonanchor

    def compress_anchor(self, anchor, scales_anchor, means_anchor, symbols_list, indexes_list):
        # squeeze anchor to avoid non-anchor symbols
        anchor_squeeze = self.ckbd_anchor_sequeeze(anchor)
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = self.gaussian_conditional.quantize(anchor_squeeze, "symbols", means_anchor_squeeze)
        symbols_list.extend(anchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat + means_anchor_squeeze)
        return anchor_hat

    def compress_nonanchor(self, nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list):
        nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(nonanchor)
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = self.gaussian_conditional.quantize(nonanchor_squeeze, "symbols", means_nonanchor_squeeze)
        symbols_list.extend(nonanchor_hat.reshape(-1).tolist())
        indexes_list.extend(indexes.reshape(-1).tolist())
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat + means_nonanchor_squeeze)
        return nonanchor_hat

    def decompress_anchor(self, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets):
        scales_anchor_squeeze = self.ckbd_anchor_sequeeze(scales_anchor)
        means_anchor_squeeze = self.ckbd_anchor_sequeeze(means_anchor)
        indexes = self.gaussian_conditional.build_indexes(scales_anchor_squeeze)
        anchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        anchor_hat = torch.Tensor(anchor_hat).reshape(scales_anchor_squeeze.shape).to(scales_anchor.device) + means_anchor_squeeze
        anchor_hat = self.ckbd_anchor_unsequeeze(anchor_hat)
        return anchor_hat

    def decompress_nonanchor(self, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets):
        scales_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(scales_nonanchor)
        means_nonanchor_squeeze = self.ckbd_nonanchor_sequeeze(means_nonanchor)
        indexes = self.gaussian_conditional.build_indexes(scales_nonanchor_squeeze)
        nonanchor_hat = decoder.decode_stream(indexes.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        nonanchor_hat = torch.Tensor(nonanchor_hat).reshape(scales_nonanchor_squeeze.shape).to(scales_nonanchor.device) + means_nonanchor_squeeze
        nonanchor_hat = self.ckbd_nonanchor_unsequeeze(nonanchor_hat)
        return nonanchor_hat
    
unsqueeze_first_two = lambda x: x.unsqueeze(0).unsqueeze(0)
from_unfold2bpchw = lambda x: x.view