
from enum import auto
from locale import normalize
from torchvision.models.resnet import ResNet
from mimetypes import init
from tkinter import Y
import torch
import torch.nn as nn
from compressai.ops.bound_ops import BoundFunction, LowerBoundFunction
import torch.nn.functional as F
from compressai.layers import (
    AttentionBlock,
    PCALayer,
    ResidualBlock,
    ConvBlock,
    ResidualBlockUpsample,
    ConvBlockUpsample,
    ResidualBlockWithStride,
    ConvBlockWithStride,
    LambdaCondition,
    LayerNorm2d,
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
from compressai.layers import MaskedConv2d
from tqdm import tqdm
import torch
# torch.use_deterministic_algorithms(True)


class JpegConditionedSequential(nn.Sequential):
    def __init__(self, *kargs, **kwargs):
        super().__init__(*tuple(filter(lambda x: x is not None, kargs)), **kwargs)
        
    def forward(self, input, x_jpg, lam_embedding=None):
        for idx, module in enumerate(self):
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, MaskedDeconv, PCALayer)) or module.__dict__.get("no_srgb", False):
                input = module(input)
            elif isinstance(module, (LambdaCondition)):
                input = module(input, lam_embedding)
            else:
                input = module(
                    torch.cat([F.interpolate(x_jpg, size=input.shape[2:]), input], dim=1))
            # input = BoundFunction.apply(input, -10, 10)
        return input

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

        weight_1x1 = self.deconv.weight[:,:,self.kernel_size//2,self.kernel_size//2].unsqueeze(2).unsqueeze(3)
        bias = self.deconv.bias
        resid = F.conv_transpose2d(input, weight_1x1, bias, padding=0)
        deconved = deconved*(1-mask) + resid*mask
        
        return torch.cat([deconved, mask], dim=1)  

class RawLearnedJournal4(Cheng2020Attention):
    def __init__(self, N=192, **kwargs):
        for useless_arg in ["noquant", "hyper_mean", "resid", "relu", "xyz", "ablation_no_raw", "demosaic", "resid_path", "context_z", "discrete_context", "channel_mean"]:
            kwargs.pop(useless_arg, False)
        self.rounding = kwargs.pop("rounding", "noise")
        self.raw_channel = kwargs.pop("raw_channel", 3)
        self.rounding_aux = kwargs.pop("rounding_aux", "noise")
        self.num_of_levels = kwargs.pop("num_of_levels", 2)
        self.stride = kwargs.pop("stride", 2)
        self.reduce_c = reduce_c = kwargs.pop("reduce_c", 1)
        self.down_num = kwargs.pop("down_num", 4)
        self.gamma = kwargs.pop("gamma", 0)
        self.gmm_num = kwargs.pop("gmm_num", None) # mixture gaussian num
        self.dis2range = kwargs.pop("dis2range", None) # The range of bins for discrete distribution
        self.drop_pixel = kwargs.pop("drop_pixel", False)
        self.soft_gumbel = kwargs.pop("soft_gumbel", False)
        
        
        # The number of sampling for context model
        self.sampling_num = kwargs.pop("sampling_num", 2)
        self.use_deconv = kwargs.pop("use_deconv", False)
        self.z_scale = kwargs.pop("z_scale", 1)
        self.multiprocess = kwargs.pop("multiprocess", False)
        self.adaptive_quant = kwargs.pop("adaptive_quant", False)
  
        self.lambda_list = kwargs.pop("lambda_list", None)
        self.pca = kwargs.pop("pca", False) # Pixel_wised_Channel_Attention
        self.drop_num = kwargs.pop("drop_num",0)
        
        embedding_dim = kwargs.pop("embedding_dim", 16)
        norm = kwargs.pop("norm", None)
        act = kwargs.pop("act", "GDN") # "lrelu") 
        if norm:
            print(f"[INFO] Using layer norm [{norm}]")
        if act!="GDN":
            print(f"[INFO] Using activation func [{act}]")
        if self.gmm_num:
            print(f"[INFO] GMM number is [{self.gmm_num}]")
        if self.soft_gumbel:
            print(f"[INFO] Using soft_gumbel for training")
        if self.rounding_aux != "noise":
            print(f"[INFO] Using {self.rounding_aux} for auxilary variable")
        self.embed_weight = True
        super().__init__(N=N, **kwargs)
        
        if self.lambda_list is not None:
            self.lambda_embeddings = nn.Embedding(len(self.lambda_list), embedding_dim=embedding_dim)
            self.lambda_map = {
                lam:i for i, lam in enumerate(self.lambda_list)
            }
        stride_num = 4
        stride_list = [self.stride if i < self.down_num else 1 for i in range(stride_num)]
        
        self.gaussian_conditional = GaussianConditional(
            None, likelihood_bound=1e-7, tail_mass=1e-9, scale_bound=0.11 if not self.adaptive_quant else 0.01)

        self.g_a_list = nn.Sequential()
        self.g_s_list = nn.Sequential()
        self.g_mask_list = nn.Sequential()
        self.entropy_parameters_list = nn.Sequential()
        self.context_prediction_list = nn.Sequential()

        for i in range(self.num_of_levels):
            if i == 0:
                in_channel = self.raw_channel + 3
            else:
                in_channel = N // reduce_c + 3
            
            
            self.mask_in_channel = N // reduce_c * 2
            g_s_out_channel = N // reduce_c * 2 if i != 0 else 3
            if i == 0:
                g_a = JpegConditionedSequential(
                    ResidualBlockWithStride(self.raw_channel+3, N, stride=stride_list[0], norm=norm, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,

                    ResidualBlock(N+3, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,

                    ResidualBlockWithStride(N+3, N, stride=stride_list[1], norm=norm, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    AttentionBlock(N+3),
                    
                    ResidualBlock(N+6, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ResidualBlockWithStride(N+3, N, stride=stride_list[2], norm=norm, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ResidualBlock(N+3, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    conv3x3(N+3, N// reduce_c, stride=stride_list[3]),
                    AttentionBlock(N// reduce_c, no_srgb=True),
                )
                
                g_s = JpegConditionedSequential(
                    AttentionBlock(N//reduce_c+3),
                    ResidualBlock(N//reduce_c + 6, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ResidualBlockUpsample(N+3, N, norm=norm, act=act, upsample=1),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ResidualBlock(N+3, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                     
                    ResidualBlockUpsample(N+3, N, norm=norm, act=act, upsample=1),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    AttentionBlock(N+3),
                    ResidualBlock(N+6, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ResidualBlockUpsample(N+3, N, norm=norm, act=act, upsample=1),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    ResidualBlock(N+3, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    subpel_conv3x3(N+3, self.raw_channel, 1),
                )
            elif i == 1:
                act = "lrelu"
                g_a = JpegConditionedSequential(
                    ConvBlock(N // reduce_c +3, N, norm=norm),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
 
                    ConvBlockWithStride(N+3, N, stride=2, norm=norm, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    conv3x3(N+3, N // reduce_c, stride=2),
                )
                g_s = JpegConditionedSequential(
                    conv3x3(N// reduce_c + 3, N),
                    nn.LeakyReLU(inplace=False),
                    
                    ConvBlockUpsample(N+3, N, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    ConvBlockUpsample(N+3, N * 3//2, act=act),
                    LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                    
                    conv3x3(N*3//2 + 3, N // reduce_c * 2),
                )
            else:
                raise NotImplementedError(f"Level is {i} not implemented")
            
            g_mask = JpegConditionedSequential(
                conv3x3(N // reduce_c * 2 + 4, N),
                nn.LeakyReLU(inplace=False),
                conv3x3(N + 3, N * 3 // 2),
                nn.LeakyReLU(inplace=False),
                conv3x3(N * 3 // 2 + 3, 2),
            )
            
            out_channel_orig = (N // reduce_c) 
            if self.gmm_num is not None:
                out_channel = out_channel_orig * 3 * self.gmm_num
            else:
                out_channel = out_channel_orig * 2
                
            if self.adaptive_quant: 
                out_channel = out_channel + out_channel_orig    
                
            entropy_parameters = JpegConditionedSequential(
                PCALayer(N // reduce_c * 4 + 1),
                nn.Conv2d(N // reduce_c * 4 + 4, N*2, 1), 
                LambdaCondition(embedding_dim, N*2) if self.lambda_list is not None else None,
                nn.LeakyReLU(inplace=False),
                
                # PCALayer(N*2),
                nn.Conv2d(N*2+3, N*2, 1),
                LambdaCondition(embedding_dim, N*2) if self.lambda_list is not None else None,
                nn.LeakyReLU(inplace=False),
                
                # PCALayer(N*2),
                nn.Conv2d(N*2+3, N*2, 1),
                LambdaCondition(embedding_dim, N*2) if self.lambda_list is not None else None,
                nn.LeakyReLU(inplace=False),
                # PCALayer(N + 3),
                # PCALayer(N*2),
                nn.Conv2d(N*2+3, out_channel, 1),
                LambdaCondition(embedding_dim, out_channel) if self.lambda_list is not None else None,
            )

            context_prediction = JpegConditionedSequential(
                MaskedDeconv(
                    in_channels=N // reduce_c * 2 + 3, out_channels=N, kernel_size=5),
                LambdaCondition(embedding_dim, N) if self.lambda_list is not None else None,
                nn.LeakyReLU(inplace=False),
                # PCALayer(N + 3),
                
                conv3x3(N + 4, N * 3 // 2),
                LambdaCondition(embedding_dim, N * 3 // 2) if self.lambda_list is not None else None,
                nn.LeakyReLU(inplace=False),
                # PCALayer(N + 3),
                
                conv3x3(N * 3 // 2 + 3, N * 2 // reduce_c),
                LambdaCondition(embedding_dim, N * 2 // reduce_c) if self.lambda_list is not None else None,
            )
            # context_prediction = JpegConditionedSequential(
            #     conv3x3(N // reduce_c * 2 + 3, N) if not self.use_deconv else MaskedDeconv(
            #         in_channels=N // reduce_c * 2, out_channels=N * 2 // reduce_c, kernel_size=3),
            # )
            
            self.g_mask_list.append(g_mask)
            self.g_a_list.append(g_a)
            self.g_s_list.append(g_s)
            self.entropy_parameters_list.append(entropy_parameters)
            self.context_prediction_list.append(context_prediction)

    def load_state_dict(self, state_dict, **kwargs):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        return super().load_state_dict(state_dict, **kwargs)

    def learned_context_entropy_bottleneck(self, y, params, x_jpg, context_prediction, gaussian_conditional: GaussianConditional, entropy_parameters, g_mask, rounding, adaptive_quant=False, use_deconv=True, lam_embedding=None, gmm_num=None, drop_pixel=False, soft=False):
        # y:
        
        sampling_num = self.sampling_num
        b, c, h, w = params.shape
        
        mask_list = []
        mask_cum_i = torch.zeros(b, 1, h, w).to(y.device)
        for i in range(sampling_num):
            x = g_mask(torch.cat([params, mask_cum_i], dim=1),x_jpg)
            mask = F.gumbel_softmax(x, hard=True, dim=1)[:,[0]]
            mask_incre = torch.maximum(mask_cum_i, mask) - mask_cum_i 
            mask_cum_i = mask_cum_i + mask_incre
            mask_list.append(mask_incre)
        # DEBUG: [mask_list[i].float().mean().item() for i in range(4)], mask_cum_i.float().mean().item()            
        ctx_params = torch.zeros_like(params)
        y_hat = torch.zeros_like(y)
        quantization_level_map = torch.zeros_like(y_hat)
        y_likelihoods = torch.zeros_like(y)
        mask_cum_i = torch.zeros(b, 1, h, w).to(y.device)
        for i in range(sampling_num):
            ctx_params = context_prediction(
                    torch.cat([(torch.cat([y_hat, quantization_level_map], dim=1) * mask_cum_i), mask_cum_i], dim=1), x_jpg, lam_embedding)
            # mask_i = (mask == i).unsqueeze(dim=1)
            
            gaussian_params_partial = entropy_parameters(
                    torch.cat((params, ctx_params, mask_cum_i), dim=1), x_jpg, lam_embedding
                )
                
            quantization_level_partial = None
            
            if gmm_num is None:
                if adaptive_quant:
                    scales_hat_partial, means_hat_partial, quantization_level_partial = gaussian_params_partial.chunk(3, 1)
                    ## Freeze quantization map
                    # quantization_level_partial = (quantization_level_partial * 0).detach()  # F.relu(quantization_level_partial, inplace=False)
                    # quantization_level_partial = BoundFunction.apply(quantization_level_partial, -3, 3)
                else:
                    scales_hat_partial, means_hat_partial = gaussian_params_partial.chunk(2, 1)
                prob_partial = None
            else:
                prob_partial_list = []
                scales_hat_partial_list = []
                means_hat_partial_list = []
                c = gaussian_params_partial.shape[1]
                if adaptive_quant:
                    quantization_level_partial = gaussian_params_partial[:,:c//(gmm_num*3+1)]
                    gaussian_params_partial = gaussian_params_partial[:,c//(gmm_num*3+1):]
                    # quantization_level_map = quantization_level_map + quantization_level_partial * mask_i
                gmm_params_partial_list = gaussian_params_partial.chunk(gmm_num, 1)
                for gmm_params_partial in gmm_params_partial_list:
                    prob_partial, scales_hat_partial, means_hat_partial = gmm_params_partial.chunk(3, 1)
                    prob_partial_list.append(prob_partial)
                    means_hat_partial_list.append(means_hat_partial)
                    scales_hat_partial_list.append(scales_hat_partial)
                prob_partial = F.softmax(torch.stack(prob_partial_list, dim=0),dim=0)
                means_hat_partial = torch.stack(means_hat_partial_list, dim=0)
                scales_hat_partial = torch.stack(scales_hat_partial_list, dim=0)
                
            y_hat_partial, y_likelihoods_partial = gaussian_conditional(y, scales_hat_partial, means=means_hat_partial, quantization_level_map=quantization_level_partial if adaptive_quant else None, prob=prob_partial, soft=soft, rounding=rounding)

            # else:
            #     c = gaussian_params_partial.shape[1]
            #     if adaptive_quant:
            #         quantization_level_map = gaussian_params_partial[:,:c//(dis2range*2+2)]
            #         gaussian_params_partial = gaussian_params_partial[:,c//(dis2range*2+2):]
            #     pre_llk = gaussian_params_partial.chunk((dis2range*2+2), 1)
            #     llk = torch.softmax(torch.stack(pre_llk, dim=0),dim=0)
            #     y_likelihoods_partial = llk
            mask_i = mask_list[i]
            if self.adaptive_quant:
                quantization_level_map = quantization_level_map + quantization_level_partial * mask_i
            if drop_pixel and i == (sampling_num-1):
                y_likelihoods += mask_i
                y_hat = y_hat + means_hat_partial * mask_i
            else:
                y_hat = y_hat + y_hat_partial * mask_i
                # symbols = symbols + torch.exp2(-quantization_level_map) * y_hat * mask_i
                y_likelihoods += y_likelihoods_partial * mask_i
                y_likelihoods = gaussian_conditional.likelihood_lower_bound(y_likelihoods)
            mask_cum_i = mask_cum_i + mask_i
            
        return y_hat, y_likelihoods

    def forward(self, x_raw, x_jpg, lam=None, soft=False):
        # y = self.g_a(x_jpg/(x_raw+1e-3) if self.embed_weight else x_raw, x_jpg)
        soft_gumbel = self.soft_gumbel and self.training
        soft = soft or soft_gumbel
        
        z_list = []
        z = x_raw
        if lam is not None:
            assert x_raw.shape[0] == 1, "Not implemented of varaible-bpp for batch size>1"
            # lam_embedding = self.lambda_embeddings(torch.tensor([self.lambda_map[l] for l in lam], device=z.device))
            lam_embedding = self.lambda_embeddings(torch.tensor([self.lambda_map[lam]], device=z.device))
        else:
            lam_embedding = None
        for i, g_a in enumerate(self.g_a_list):
            z = g_a(z, x_jpg, lam_embedding)
            z_list.append(z)
            # z = F.interpolate(z, scale_factor=0.5) # TODO: remove this

        
        likelihoods = {}
        params = torch.zeros_like(z_list[-1])[:,[0]].repeat(1, self.mask_in_channel, 1, 1)
        for idx, (g_s, z, context_prediction, entropy_parameters, g_mask) in enumerate(reversed(list(zip(self.g_s_list, z_list, self.context_prediction_list, self.entropy_parameters_list, self.g_mask_list)))):
            z_hat, z_likelihoods = self.learned_context_entropy_bottleneck(
                z, params, x_jpg, context_prediction, self.gaussian_conditional, entropy_parameters, g_mask, self.rounding if idx==1 else self.rounding_aux, use_deconv=self.use_deconv, adaptive_quant=self.adaptive_quant, lam_embedding=lam_embedding, gmm_num=self.gmm_num, drop_pixel=self.drop_pixel, soft=soft)
            likelihoods[f"level_{idx}"] = z_likelihoods
            if idx == 1 and self.down_num > 0 and self.stride > 1:
                z_hat = F.interpolate(z_hat, scale_factor=self.stride**self.down_num, mode="bicubic")
            params = g_s(z_hat, x_jpg, lam_embedding)
            # if idx != (self.num_of_levels-1):
            #     params = F.interpolate(params, scale_factor=2)
        x_hat = params
        
        res_dict = {}
        res_dict.update({
            "x_hat": x_hat,
            "likelihoods": likelihoods
        })
        return res_dict

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table(min=0.11 if not self.adaptive_quant else 0.01, max=32)
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x_raw, x_jpg, **kwargs):
        if self.lambda_list is not None:
            lmbda = kwargs.get("lmbda", None)
            assert isinstance(lmbda, (float, int))
            lmbda_embedding = self.lambda_embeddings(torch.tensor([self.lambda_map[lmbda]], device=x_raw.device))
        else:
            lmbda_embedding = None
        z_list = []
        z = x_raw
        for i, g_a in enumerate(self.g_a_list):
            z = g_a(z, x_jpg, lmbda_embedding)
            z_list.append(z)
            # z = F.interpolate(z, scale_factor=0.5)
            
        params = torch.zeros_like(z_list[-1])[:,[0]].repeat(1, self.mask_in_channel, 1, 1)
        initial_shape = params.shape
        strings_list = []
        mask_list = []
        z_shape_list = []
        debug_z_hat = []
        debug_params = []
        for idx, (g_s, z, context_prediction, entropy_parameters, g_mask) in enumerate(reversed(list(zip(self.g_s_list, z_list, self.context_prediction_list, self.entropy_parameters_list, self.g_mask_list)))):
            z_strings = []
            b, c, h, w = z.shape
            
            sampling_num = self.sampling_num
            mask_final = torch.ones(b, 1, h, w).to(z.device) * sampling_num
            mask_cum_i = torch.zeros(b, 1, h, w).to(z.device)
            for i in range(sampling_num):
                x = g_mask(torch.cat([params, mask_cum_i], dim=1),x_jpg)
                mask = F.gumbel_softmax(x, hard=True, dim=1)[:,[0]]
                mask_incre = torch.maximum(mask_cum_i, mask) - mask_cum_i 
                mask_cum_i = mask_cum_i + mask_incre
                mask_index = mask_incre.nonzero(as_tuple=True)
                mask_final[:,:,mask_index[2],mask_index[3]] = i
            
            # mask_final = mask_final*0 
            # mask_final = 3 - mask_final
            
            quantization_level_map = torch.zeros_like(z)

            mask_list.append(mask_final)
            ctx_params = torch.zeros_like(params)
            z_hat = torch.zeros_like(z)
            
            quantized_value_for_debug = torch.zeros_like(z, dtype=torch.float16)
            means_hat_for_debug = torch.zeros_like(z, dtype=torch.float16)
            z_shape_list.append(z_hat.shape)
            # z_likelihoods = torch.zeros_like(z)
            # mask_cum_i = torch.zeros(b, 1, h, w).to(z.device)
            for i in range(sampling_num):
                if self.use_deconv:
                    ctx_params = context_prediction(torch.cat([z_hat, quantization_level_map, mask_final<i], dim=1), x_jpg, lmbda_embedding)
                else:
                    ctx_params = ctx_params + context_prediction(torch.cat([z_hat, mask_i], dim=1), x_jpg, lmbda_embedding)
                mask_i = (mask_final == i)
                mask_index = torch.nonzero(mask_i,as_tuple=True)

                gaussian_params_partial = entropy_parameters(
                    torch.cat((params, ctx_params, mask_final<i), dim=1), x_jpg, lmbda_embedding
                )
                
            
                quantization_level_partial = None
                if mask_i.sum() == 0:
                    string = [""]
                else:
                    if self.gmm_num is None:
                        if self.adaptive_quant:
                            scales_hat_partial, means_hat_partial, quantization_level_partial = gaussian_params_partial.chunk(3, 1)
                            if idx >= (1 - self.drop_num // sampling_num) and (sampling_num - i) <= (self.drop_num % sampling_num + 1):
                                quantization_level_partial_ = self.quan_func(quantization_level_partial)
                            else:
                                quantization_level_partial_ = quantization_level_partial
                            quantization_level_map = quantization_level_map + quantization_level_partial * mask_i 
                        else:
                            scales_hat_partial, means_hat_partial = gaussian_params_partial.chunk(2, 1)
                        if idx==1 and (self.sampling_num - i) <= self.drop_num:
                            string = [""]
                        else:
                            indexes = self.gaussian_conditional.build_indexes(scales_hat_partial[:,:,mask_index[2],mask_index[3]], quantization_level_map=quantization_level_partial_[:,:,mask_index[2],mask_index[3]] if self.adaptive_quant else None)
                            
                            indexes = indexes.int()
                            string = self.gaussian_conditional.compress(z[:,:,mask_index[2],mask_index[3]], indexes, means_hat_partial[:,:,mask_index[2],mask_index[3]], multiprocess=self.multiprocess, quantization_level_map=quantization_level_partial_[:,:,mask_index[2],mask_index[3]] if quantization_level_partial is not None else None)
                            
                    else:
                        prob_partial_list = []
                        scales_hat_partial_list = []
                        means_hat_partial_list = []
                        c = gaussian_params_partial.shape[1]
                        if self.adaptive_quant:
                            quantization_level_partial = gaussian_params_partial[:,:c//(self.gmm_num*3+1)]
                            # if idx >= (1 - self.drop_num // sampling_num) and (sampling_num - i) <= (self.drop_num % sampling_num + 1):
                            #     quantization_level_partial = self.quan_func(idx, i, quantization_level_partial)
                            gaussian_params_partial = gaussian_params_partial[:,c//(self.gmm_num*3+1):]
                            
                        gmm_params_partial_list = gaussian_params_partial.chunk(self.gmm_num, 1)
                        for gmm_params_partial in gmm_params_partial_list:
                            prob_partial, scales_hat_partial, means_hat_partial = gmm_params_partial.chunk(3, 1)
                            prob_partial_list.append(prob_partial)
                            means_hat_partial_list.append(means_hat_partial)
                            scales_hat_partial_list.append(scales_hat_partial)
                        prob_partial = F.softmax(torch.stack(prob_partial_list, dim=0),dim=0)
                        means_hat_partial = torch.stack(means_hat_partial_list, dim=0)
                        scales_hat_partial = torch.stack(scales_hat_partial_list, dim=0)
                        string = self.gaussian_conditional.compress_gmm(z[:,:,mask_index[2],mask_index[3]], means_hat_partial[:,:,:,mask_index[2],mask_index[3]], scales_hat_partial[:,:,:,mask_index[2],mask_index[3]], prob_partial[:,:,:,mask_index[2],mask_index[3]], quantization_level_partial[:,:,mask_index[2],mask_index[3]])

                    
                    # z_hat_partial = self.gaussian_conditional.quantize(z, "symbols", means=means_hat_partial) + means_hat_partial
                    # z_hat = z_hat + z_hat_partial * mask_i
                    if quantization_level_partial is not None:
                        mutiplier = torch.exp2(quantization_level_partial[:,:,mask_index[2],mask_index[3]])
                    else:
                        mutiplier = 1
                        
                    if len(means_hat_partial.shape) != 5: # not GMM
                        if idx==1 and (self.sampling_num - i) <= self.drop_num:
                            quantized_value = means_hat_partial[:,:,mask_index[2],mask_index[3]]
                            z_hat[:,:,mask_index[2],mask_index[3]] = quantized_value
                        else:
                            quantized_value = self.gaussian_conditional.quantize(z[:,:,mask_index[2],mask_index[3]], "symbols", means=means_hat_partial[:,:,mask_index[2],mask_index[3]], quantization_level_map=quantization_level_partial_[:,:,mask_index[2],mask_index[3]] if quantization_level_partial is not None else None)
                            z_hat[:,:,mask_index[2],mask_index[3]] = (quantized_value * mutiplier + means_hat_partial[:,:,mask_index[2],mask_index[3]]).half()
                    else:
                        qmap = quantization_level_partial[:,:,mask_index[2],mask_index[3]] if quantization_level_partial is not None else None
                        quantized_value = self.gaussian_conditional.quantize(z[:,:,mask_index[2],mask_index[3]], "symbols", quantization_level_map=qmap)
                        z_hat[:,:,mask_index[2],mask_index[3]] = (quantized_value * mutiplier)
                        
                    quantized_value_for_debug[:,:,mask_index[2],mask_index[3]] = quantized_value.half()
                    means_hat_for_debug[:,:,mask_index[2],mask_index[3]] = means_hat_partial[:,:,mask_index[2],mask_index[3]].half()
                 # equal to decompress
                # assert (z_hat[:,:,mask_index[2],mask_index[3]]  != self.gaussian_conditional.decompress(string, indexes, means=means_hat_partial[:,:,mask_index[2],mask_index[3]], quantization_level_map=quantization_level_map[:,:,mask_index[2],mask_index[3]])).sum() ==0

                z_strings.append(string)
                # debug_params.append([indexes.detach().cpu(), gaussian_params_partial.detach().cpu(), ctx_params.detach().cpu(), params.detach().cpu()])nn
            debug_z_hat.append(z_hat.detach().cpu())
            # 
            if idx != (self.num_of_levels-1):
                params = g_s(z_hat, x_jpg, lmbda_embedding)
            #     params = F.interpolate(params, scale_factor=2)            
            strings_list.append(z_strings)
        res = {"strings": strings_list, "shape":initial_shape, "mask_list": mask_list, "z_shape_list": z_shape_list, "debug_z_hat": debug_z_hat, "debug_params":debug_params}
        return res
    # def iter_str_length(list_x):
    #     string_length = []
    #     for x in list_x:
    #         if isinstance(x, (str, bytes)):
    #             string_length.append(len(x))
    #         else:
    #             string_length.extend(iter_str_length(x))
    #     return string_length
    def quan_func(self, map):
        # map = F.relu(map)
        # map *= 0 
        return map + self.gamma
    
    def decompress(self, strings, shape, x_jpg, compressed_result=None, **kwargs):
        assert isinstance(strings, list)  # and len(strings) == 2
        mask_list = compressed_result["mask_list"]
        z_shape_list = compressed_result["z_shape_list"]
        params = torch.zeros(shape).to(x_jpg.device)
        
        if self.lambda_list is not None:
            lmbda = kwargs.get("lmbda", None)
            assert isinstance(lmbda, (float, int))
            lmbda_embedding = self.lambda_embeddings(torch.tensor([self.lambda_map[lmbda]], device=x_jpg.device))
        else:
            lmbda_embedding = None
            
        for idx, (g_s, context_prediction, entropy_parameters, mask, string_list, z_shape) in enumerate(reversed(list(zip(self.g_s_list, self.context_prediction_list, self.entropy_parameters_list, reversed(mask_list), reversed(strings), reversed(z_shape_list))))):
            z_hat = torch.zeros(z_shape).half().to(x_jpg.device)
            ctx_params = torch.zeros_like(params)
            quantization_level_map = torch.zeros_like(z_hat)    
            for i in range(self.sampling_num):
                if self.use_deconv:
                    ctx_params = context_prediction(torch.cat([z_hat, quantization_level_map, mask<i], dim=1), x_jpg, lmbda_embedding)
                else:
                    ctx_params = ctx_params + context_prediction(torch.cat([z_hat, mask_i], dim=1), x_jpg, lmbda_embedding)
                mask_i = (mask == i)
                
                gaussian_params_partial = entropy_parameters(
                    torch.cat((params, ctx_params, mask<i), dim=1), x_jpg, lmbda_embedding
                )

            
                mask_index = torch.nonzero(mask_i,as_tuple=True)
                if self.gmm_num is None:
                    if self.adaptive_quant:
                        scales_hat_partial, means_hat_partial, quantization_level_partial = gaussian_params_partial.chunk(3, 1)
                        sampling_num = self.sampling_num
                        if idx >= (1 - self.drop_num // sampling_num) and (sampling_num - i) <= (self.drop_num % sampling_num + 1):
                            quantization_level_partial_ = self.quan_func(quantization_level_partial)
                        else:
                            quantization_level_partial_ = quantization_level_partial  
                        quantization_level_map = quantization_level_map + quantization_level_partial * mask_i 
                        # quantization_level_map *= 0.7
                        # quantization_level_map += 2
                    else:
                        scales_hat_partial, means_hat_partial = gaussian_params_partial.chunk(2, 1)
                        quantization_level_partial = None
                    if idx==1 and (self.sampling_num - i) <= self.drop_num:
                        z_hat_partial = means_hat_partial[:,:,mask_index[2],mask_index[3]]
                    else:
                        indexes = self.gaussian_conditional.build_indexes(scales_hat_partial[:,:,mask_index[2], mask_index[3]], quantization_level_map=quantization_level_partial_[:,:,mask_index[2], mask_index[3]] if self.adaptive_quant else None)
                        z_hat_partial = self.gaussian_conditional.decompress(string_list[i], indexes, means=means_hat_partial[:,:,mask_index[2],mask_index[3]], quantization_level_map=quantization_level_partial_[:,:,mask_index[2],mask_index[3]] if quantization_level_partial is not None else None)
                else:
                    prob_partial_list = []
                    scales_hat_partial_list = []
                    means_hat_partial_list = []
                    c = gaussian_params_partial.shape[1]
                    if self.adaptive_quant:
                        quantization_level_partial = gaussian_params_partial[:,:c//(self.gmm_num*3+1)]
                        if idx == 1:
                            quantization_level_partial = self.quan_func(idx, i, quantization_level_partial)
                        gaussian_params_partial = gaussian_params_partial[:,c//(self.gmm_num*3+1):]
                        
                    gmm_params_partial_list = gaussian_params_partial.chunk(self.gmm_num, 1)
                    for gmm_params_partial in gmm_params_partial_list:
                        prob_partial, scales_hat_partial, means_hat_partial = gmm_params_partial.chunk(3, 1)
                        prob_partial_list.append(prob_partial)
                        means_hat_partial_list.append(means_hat_partial)
                        scales_hat_partial_list.append(scales_hat_partial)
                    prob_partial = F.softmax(torch.stack(prob_partial_list, dim=0),dim=0)
                    means_hat_partial = torch.stack(means_hat_partial_list, dim=0)
                    scales_hat_partial = torch.stack(scales_hat_partial_list, dim=0)
                    qmap_partial = quantization_level_partial[:,:,mask_index[2],mask_index[3]] if quantization_level_partial is not None else None
                    z_hat_partial = self.gaussian_conditional.decompress_gmm(string_list[i], means=means_hat_partial[:,:,:,mask_index[2],mask_index[3]], scales=scales_hat_partial[:,:,:,mask_index[2],mask_index[3]], quantization_level_map=qmap_partial, probs=prob_partial[:,:,:,mask_index[2],mask_index[3]])



                if z_hat_partial.isinf().sum()>0:
                    print("[WARNING] %d inf in z_hat_partial (%d, %d)"%(z_hat_partial.isinf().sum(), idx, i))
                    z_hat_partial[z_hat_partial==float("inf")] = 0
                    
                z_hat[:,:,mask_index[2],mask_index[3]] = z_hat_partial.half()
            if idx == 1 and self.down_num > 0 and self.stride > 1:
                z_hat = F.interpolate(z_hat, scale_factor=self.stride**self.down_num, mode="bicubic")
            params = g_s(z_hat, x_jpg, lmbda_embedding)   
            # if idx != (self.num_of_levels-1):
            #     params = F.interpolate(params, scale_factor=2)
        x_hat = params
        return {"x_hat": x_hat}

def unsqueeze_first_two(x): return x.unsqueeze(0).unsqueeze(0)
def from_unfold2bpchw(x): return x.view
