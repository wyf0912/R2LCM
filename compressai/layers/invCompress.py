import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import warnings
from compressai.layers import AttentionBlock
import scipy
import torch
from . import thops

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels, LU_decomposed=True):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        if not LU_decomposed:
            # Sample a random orthogonal matrix:
            self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        else:
            np_p, np_l, np_u = scipy.linalg.lu(w_init)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(np.abs(np_s))
            np_u = np.triu(np_u, k=1)
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)
            eye = np.eye(*w_shape, dtype=np.float32)

            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))
            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))
            self.l_mask = torch.Tensor(l_mask)
            self.eye = torch.Tensor(eye)
        self.w_shape = w_shape
        self.LU = LU_decomposed

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not self.LU:
            pixels = thops.pixels(input)
            dlogdet = torch.slogdet(self.weight)[1] * pixels
            if not reverse:
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
            else:
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)
            return weight, dlogdet
        else:
            self.p = self.p.to(input.device)
            self.sign_s = self.sign_s.to(input.device)
            self.l_mask = self.l_mask.to(input.device)
            self.eye = self.eye.to(input.device)
            l = self.l * self.l_mask + self.eye
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))
            dlogdet = thops.sum(self.log_s) * thops.pixels(input)
            if not reverse:
                w = torch.matmul(self.p, torch.matmul(l, u))
            else:
                l = torch.inverse(l.double()).float()
                u = torch.inverse(u.double()).float()
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1), dlogdet

    def forward(self, input, logdet=None, reverse=False):
        """
        log-det = log|abs(|W|)| * pixels
        """
        weight, dlogdet = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet + dlogdet
            return z # , logdet
        else:
            z = F.conv2d(input, weight)
            if logdet is not None:
                logdet = logdet - dlogdet
            return z # , logdet

class AttModule(nn.Module):
    def __init__(self, N):
        super(AttModule, self).__init__()
        self.forw_att = AttentionBlock(N)
        self.back_att = AttentionBlock(N)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_att(x)
        else:
            return self.back_att(x)

class EnhModule(nn.Module):
    def __init__(self, nf):
        super(EnhModule, self).__init__()
        self.forw_enh = EnhBlock(nf)
        self.back_enh = EnhBlock(nf)

    def forward(self, x, rev=False):
        if not rev:
            return self.forw_enh(x)
        else:
            return self.back_enh(x)

class EnhBlock(nn.Module):
    def __init__(self, nf):
        super(EnhBlock, self).__init__()
        self.layers = nn.Sequential(
            DenseBlock(16, nf),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=True),
            DenseBlock(nf, 16)
        )

    def forward(self, x):
        return x + self.layers(x) * 0.2

class InvComp(nn.Module):
    def __init__(self, M):
        super(InvComp, self).__init__()
        self.in_nc = 16
        self.out_nc = M
        self.operations = nn.ModuleList()

        # 1st level
        b = SqueezeLayer(2)
        self.operations.append(b)
        self.in_nc *= 4
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)

        # 2nd level
        b = SqueezeLayer(2)
        self.operations.append(b)
        self.in_nc *= 4
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 5)
        self.operations.append(b)

        # 3rd level
        b = SqueezeLayer(2)
        self.operations.append(b)
        self.in_nc *= 4
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations.append(b)

        # 4th level
        b = SqueezeLayer(2)
        self.operations.append(b)
        self.in_nc *= 4
        b = InvertibleConv1x1(self.in_nc)
        self.operations.append(b)
        b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        self.operations.append(b)
        # b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        # self.operations.append(b)
        # b = CouplingLayer(self.in_nc // 4, 3 * self.in_nc // 4, 3)
        # self.operations.append(b)

    def forward(self, x, rev=False):
        if not rev:
            for op in self.operations:
                x = op.forward(x, False)
            b, c, h, w = x.size()
            x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
        else:
            times = self.in_nc // self.out_nc
            x = x.repeat(1, times, 1, 1)
            for op in reversed(self.operations):
                x = op.forward(x, True)
        return x

class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return torch.cat((y1, y2), 1)

class CondAffineSeparatedAndCond(nn.Module):
    def __init__(self, in_channels, opt):
        super().__init__()
        self.need_features = True
        self.in_channels = in_channels
        self.in_channels_rrdb = opt_get(opt, ['network_G', 'flow', 'conditionInFeaDim'], 320)
        self.kernel_hidden = 1
        self.n_hidden_layers = 1
        hidden_channels = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'hidden_channels'])
        self.hidden_channels = 64 if hidden_channels is None else hidden_channels

        self.affine_eps = opt_get(opt, ['network_G', 'flow', 'CondAffineSeparatedAndCond', 'eps'], 0.001)

        self.channels_for_nn = self.in_channels // 2
        self.channels_for_co = self.in_channels - self.channels_for_nn

        if self.channels_for_nn is None:
            self.channels_for_nn = self.in_channels // 2

        self.fAffine = self.F(in_channels=self.channels_for_nn + self.in_channels_rrdb,
                              out_channels=self.channels_for_co * 2,
                              hidden_channels=self.hidden_channels,
                              kernel_hidden=self.kernel_hidden,
                              n_hidden_layers=self.n_hidden_layers)

        self.fFeatures = self.F(in_channels=self.in_channels_rrdb,
                                out_channels=self.in_channels * 2,
                                hidden_channels=self.hidden_channels,
                                kernel_hidden=self.kernel_hidden,
                                n_hidden_layers=self.n_hidden_layers)
        self.opt = opt
        self.le_curve = opt['le_curve'] if opt['le_curve'] is not None else False
        if self.le_curve:
            self.fCurve = self.F(in_channels=self.in_channels_rrdb,
                                 out_channels=self.in_channels,
                                 hidden_channels=self.hidden_channels,
                                 kernel_hidden=self.kernel_hidden,
                                 n_hidden_layers=self.n_hidden_layers)

    def forward(self, input: torch.Tensor, logdet=None, reverse=False, ft=None, attention_map=None):
        if attention_map is not None: 
            if opt_get(self.opt, ["invMulFlow"], False):
                ft = ft * attention_map
            else:
                ft = ft * (1-attention_map+0.015)
        if not reverse:
            z = input
            assert z.shape[1] == self.in_channels, (z.shape[1], self.in_channels)

            # gray map attention
            # if attention_map is not None:
            #     z = z / attention_map
            #     logdet = logdet - self.get_logdet(attention_map) * z.shape[1]

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)

            # save_image(unsqueeze2d(torch.cat([self.feature_extract(ft, self.fFeatures)[0], self.feature_extract(ft * (1-attention_map), self.fFeatures)[0]], dim=0), 2), 'd.jpg', normalize=True)
            # save_image(unsqueeze2d(torch.cat([self.feature_extract(ft, self.fFeatures)[1], self.feature_extract(ft * (1-attention_map), self.fFeatures)[1]], dim=0), 2), 'd2.jpg', normalize=True)

            z = z + shiftFt # ((shiftFt / attention_map) if attention_map is not None else shiftFt)
            if not opt_get(self.opt, ['NoScale'], False):
                if opt_get(self.opt, ["invMulFlow"], False):
                    z = z /  ((scaleFt * attention_map) if attention_map is not None else scaleFt)
                    logdet = logdet - self.get_logdet((scaleFt * attention_map) if attention_map is not None else scaleFt)
                else:
                    z = z * ((scaleFt / attention_map) if attention_map is not None else scaleFt)
                    logdet = logdet + self.get_logdet((scaleFt / attention_map) if attention_map is not None else scaleFt)

            # Curve conditional
            if self.le_curve:
                # logdet = logdet + thops.sum(torch.log(torch.sigmoid(z) * (1 - torch.sigmoid(z))), dim=[1, 2, 3])
                # z = torch.sigmoid(z)
                # alpha = self.fCurve(ft)
                # alpha = (torch.tanh(alpha + 2.) + self.affine_eps)
                # logdet = logdet + thops.sum(torch.log((1 + alpha - 2 * z * alpha).abs()), dim=[1, 2, 3])
                # z = z + alpha * z * (1 - z)

                alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                alpha = torch.relu(alpha) + self.affine_eps
                logdet = logdet + thops.sum(torch.log(alpha * torch.pow(z.abs(), alpha - 1)) + self.affine_eps)
                z = torch.pow(z.abs(), alpha) * z.sign()

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            z2 = z2 + shift # ((shift / attention_map) if attention_map is not None else shift)
            if not opt_get(self.opt, ['NoScale'], False):
                if opt_get(self.opt, ["invMulFlow"], False):
                    z2 = z2 / scale
                    logdet = logdet - self.get_logdet(scale)
                else:
                    z2 = z2 * scale # ((scale / attention_map) if attention_map is not None else scale)
                    logdet = logdet + self.get_logdet(scale)
                # if attention_map is not None:
                #     logdet = logdet - self.get_logdet(attention_map) * z2.shape[1]
            z = thops.cat_feature(z1, z2)
            output = z
        else:
            z = input

            # Self Conditional
            z1, z2 = self.split(z)
            scale, shift = self.feature_extract_aff(z1, ft, self.fAffine)
            self.asserts(scale, shift, z1, z2)
            if not opt_get(self.opt, ['NoScale'], False):
                if opt_get(self.opt, ["invMulFlow"], False):
                    z2 = z2 * scale
                else:
                    z2 = z2 / scale # / ((scale * attention_map) if attention_map is not None else scale)
            z2 = z2 - shift # - ((shift * attention_map) if attention_map is not None else shift)
            z = thops.cat_feature(z1, z2)
            # logdet = logdet - self.get_logdet(scale)
            # if attention_map is not None:
            #     logdet = logdet + self.get_logdet(attention_map) * z2.shape[1]

            # Curve conditional
            if self.le_curve:
                # alpha = self.fCurve(ft)
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                # z = (1 + alpha) / alpha - (
                #             alpha + torch.pow(2 * alpha - 4 * alpha * z + torch.pow(alpha, 2) + 1, 0.5) + 1) / (
                #             2 * alpha)
                # z = torch.log((z / (1 - z)).clamp(1 / 1000, 1000))

                alpha = self.fCurve(ft)
                alpha = torch.relu(alpha) + self.affine_eps
                # alpha = (torch.sigmoid(alpha + 2.) + self.affine_eps)
                z = torch.pow(z.abs(), 1 / alpha) * z.sign()

            # Feature Conditional
            scaleFt, shiftFt = self.feature_extract(ft, self.fFeatures)
            if not opt_get(self.opt, ['NoScale'], False):
                if opt_get(self.opt, ["invMulFlow"], False):
                    z = z * scaleFt
                    logdet = logdet + self.get_logdet(scaleFt)
                else:
                    z = z / scaleFt # ((scaleFt * attention_map) if attention_map is not None else scaleFt)
                    logdet = logdet - self.get_logdet(scaleFt)
            z = z - shiftFt # ((shiftFt * attention_map) if attention_map is not None else shiftFt)

            # gray map attention
            # if attention_map is not None:
            #     z = z * attention_map
            #     logdet = logdet + self.get_logdet(attention_map) * z.shape[1]
            output = z
        return output, logdet

    def asserts(self, scale, shift, z1, z2):
        assert z1.shape[1] == self.channels_for_nn, (z1.shape[1], self.channels_for_nn)
        assert z2.shape[1] == self.channels_for_co, (z2.shape[1], self.channels_for_co)
        assert scale.shape[1] == shift.shape[1], (scale.shape[1], shift.shape[1])
        assert scale.shape[1] == z2.shape[1], (scale.shape[1], z1.shape[1], z2.shape[1])

    def get_logdet(self, scale):
        return thops.sum(torch.log(scale), dim=[1, 2, 3])

    def feature_extract(self, z, f):
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = self.scale_act_func(scale)
        return scale, shift

    def feature_extract_aff(self, z1, ft, f):
        # ## DEBUG
        # max_val = z1.abs().max()
        # if max_val>20:
        #     print(max_val)
        # z1 = z1.clamp(-20,20)
        # ## END DEBUG
        z = torch.cat([z1, ft], dim=1)
        h = f(z)
        shift, scale = thops.split_feature(h, "cross")
        scale = self.scale_act_func(scale)
        return scale, shift

    def scale_act_func(self, scale):
        if opt_get(self.opt, ["flowNoSigmoid"], False):
            if opt_get(self.opt, ["minimum_one"], False):
                return F.relu(scale)+ 1
            return F.relu(scale+1)+ 0.2
        else:
            if opt_get(self.opt, ["invMulFlow"], False):
                scale = (torch.sigmoid(scale + 2.) + 1e-1)
            else:
                scale = (torch.sigmoid(scale + 2.) + self.affine_eps)
            return scale

    def split(self, z):
        z1 = z[:, :self.channels_for_nn]
        z2 = z[:, self.channels_for_nn:]
        assert z1.shape[1] + z2.shape[1] == z.shape[1], (z1.shape[1], z2.shape[1], z.shape[1])
        return z1, z2

    def F(self, in_channels, out_channels, hidden_channels, kernel_hidden=1, n_hidden_layers=1):
        layers = [Conv2d(in_channels, hidden_channels), nn.ReLU(inplace=False)]

        for _ in range(n_hidden_layers):
            layers.append(Conv2d(hidden_channels, hidden_channels, kernel_size=[kernel_hidden, kernel_hidden]))
            layers.append(nn.ReLU(inplace=False))
        layers.append(Conv2dZeros(hidden_channels, out_channels))

        return nn.Sequential(*layers)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)
        
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input, reverse=False):
        if not reverse:
            output = self.squeeze2d(input, self.factor)  # Squeeze in forward
            return output
        else:
            output = self.unsqueeze2d(input, self.factor)
            return output
        
    def jacobian(self, x, rev=False):
        return 0
        
    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

class InvertibleConv1x1Vanilla(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input, reverse=False):
        weight = self.get_weight(input, reverse)
        if not reverse:
            z = F.conv2d(input, weight)
            return z
        else:
            z = F.conv2d(input, weight)
            return z

class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu", train=False):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block_simplified, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = IRes_Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape
            
        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, padding=0),
                                                  (in_ch, h, w), kernel_size1))
        layers.append(nonlin())
        kernel_size3 = 1
        layers.append(self._wrapper_spectral_norm(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=0),
                                                  (int_ch, h, w), kernel_size3))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = ActNorm2D(in_ch, train=train)
        else:
            self.actnorm = None

    def forward(self, x, rev=False, ignore_logdet=False, maxIter=25):
        if not rev:
            """ bijective or injective block forward """
            if self.stride == 2:
                x = self.squeeze.forward(x)
            if self.actnorm is not None:
                x, an_logdet = self.actnorm(x)
            else:
                an_logdet = 0.0
            Fx = self.bottleneck_block(x)
            if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
                trace = torch.tensor(0.)
            else:
                trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)
            y = Fx + x
            return y, trace + an_logdet
        else:
            y = x
            for iter_index in range(maxIter):
                summand = self.bottleneck_block(x)
                x = y - summand

            if self.actnorm is not None:
                x = self.actnorm.inverse(x)
            if self.stride == 2:
                x = self.squeeze.inverse(x)
            return x
    
    def _wrapper_spectral_norm(self, layer, shapes, kernel_size):
        if kernel_size == 1:
            # use spectral norm fc, because bound are tight for 1x1 convolutions
            return spectral_norm_fc(layer, self.coeff, 
                                    n_power_iterations=self.n_power_iter)
        else:
            # use spectral norm based on conv, because bound not tight
            return spectral_norm_conv(layer, self.coeff, shapes,
                                      n_power_iterations=self.n_power_iter)