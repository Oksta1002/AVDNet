import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt, sin, cos, pi
from vd.utils.registry import ARCH_REGISTRY
from vd.archs.arch_util import DCNv2Pack


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, norm = False):
        super(EncoderBlock, self).__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.InstanceNorm2d(in_channels),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class MoireEstimation(nn.Module):
    def __init__(self, in_channels):
        super(MoireEstimation, self).__init__()
        self.enc1 = EncoderBlock(in_channels, norm=True)
        self.enc2 = EncoderBlock(in_channels)
        self.enc3 = EncoderBlock(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, in_channels)
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.pool(x).squeeze()
        x = self.tanh(self.fc(x))
        return x


class DilatedBlock(nn.Module):
    def __init__(self, in_channel, d_list, num_feat):
        super(DilatedBlock, self).__init__()
        self.d_list = d_list
        self.conv_layers = nn.ModuleList()
        c = in_channel
        for i in range(len(d_list)):
            dense_conv = nn.Sequential(nn.Conv2d(in_channels=c, out_channels=num_feat, kernel_size=3, dilation=d_list[i],
                                   padding=d_list[i]),
                                   nn.ReLU(inplace=True)
            )
            self.conv_layers.append(dense_conv)
            c = c + num_feat
        self.conv_post = nn.Conv2d(in_channels=c, out_channels=in_channel, kernel_size=1, padding=0)

    def forward(self, x):
        t = x
        for conv_layer in self.conv_layers:
            _t = conv_layer(t)
            t = torch.cat([_t, t], dim=1)
        t = self.conv_post(t)
        return t


class AdaptiveBandpassFilter(nn.Module):
    def __init__(self):
        super(AdaptiveBandpassFilter, self).__init__()
        self.after_trans = nn.Conv2d(64, 64, 3, 1, 1)
        self.weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.register_buffer('kernel', self.initialize_kernel())
        
    def initialize_kernel(self):
        conv_shape = (64, 64, 1, 1)
        kernel = torch.zeros(conv_shape)
        r1 = sqrt(1.0 / 8)
        r2 = sqrt(2.0 / 8)
        for i in range(8):
            _u = 2 * i + 1
            for j in range(8):
                _v = 2 * j + 1
                index = i * 8 + j
                for u in range(8):
                    for v in range(8):
                        index2 = u * 8 + v
                        t = cos(_u * u * pi / 16) * cos(_v * v * pi / 16)
                        t = t * r1 if u == 0 else t * r2
                        t = t * r1 if v == 0 else t * r2
                        kernel[index2, index, 0, 0] = t
        return kernel

    def forward(self, x, params):
        N, C, H, W = x.size()
        params = params.reshape(64 * N, 1, 1, 1)
        f = x.reshape(1, N * C, H, W)
        kernel = self.kernel.repeat(N,1,1,1) * params
        f_t = F.conv2d(input=f, weight=kernel, padding='same', groups=N)
        f_t = f_t.reshape(N, C, H, W)
        f = self.after_trans(f_t)
        x = x + f * self.weight
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, 1)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.pool(x)
        y = self.relu(self.conv1(y))
        y = self.relu(self.conv2(y))
        y = self.sigmoid(self.conv3(y))
        return y


class ABB(nn.Module):
    def __init__(self, in_channels, num_feat):
        super(ABB, self).__init__()
        self.mpe = MoireEstimation(in_channels)
        self.db = DilatedBlock(in_channel=in_channels, d_list=(1, 2, 3, 2, 1), num_feat=num_feat)
        self.abf = AdaptiveBandpassFilter()
        self.attention = ChannelAttention(in_channels)
        
    def forward(self, x):
        theta = self.mpe(x)
        feat1 = self.db(x)
        feat2 = self.abf(feat1, theta)
        feat3 = feat2 * self.attention(feat2)
        return x + feat3


class SpatialAttention(nn.Module):
    def __init__(self, in_channels=64):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels // 4, 7, 1, 3)
        self.conv3 = nn.Conv2d(in_channels // 4, in_channels, 1)
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        feat = self.leaky_relu(self.conv1(x))
        feat = self.leaky_relu(self.conv2(feat))
        feat = self.sigmoid(self.conv3(feat))
        return feat * x


class SGAB(nn.Module):
    def __init__(self, num_feat=64, deformable_groups=8):
        super(SGAB, self).__init__()
        self.offset_conv1 = nn.ModuleDict()
        self.offset_conv2 = nn.ModuleDict()
        self.offset_conv3 = nn.ModuleDict()
        self.diff_conv = nn.ModuleDict()
        self.spatial_att = nn.ModuleDict()
        self.dcn_pack = nn.ModuleDict()
        self.feat_conv = nn.ModuleDict()
        for i in range(3, 0, -1):
            level = f'l{i}'
            self.offset_conv1[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            self.offset_conv2[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if i == 3:
                self.offset_conv3[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            else:
                self.offset_conv3[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
            self.diff_conv[level] = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.spatial_att[level] = SpatialAttention(num_feat)
            self.dcn_pack[level] = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=deformable_groups)

            if i < 3:
                self.feat_conv[level] = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat_l, ref_feat_l):
        upsampled_offset, upsampled_feat = None, None
        for i in range(3, 0, -1):
            level = f'l{i}'
            offset = torch.cat([nbr_feat_l[i - 1], ref_feat_l[i - 1]], dim=1)
            offset = self.lrelu(self.offset_conv1[level](offset))
            offset = self.lrelu(self.offset_conv2[level](offset))
            diff = ref_feat_l[i - 1] - nbr_feat_l[i - 1]
            diff = self.lrelu(self.diff_conv[level](diff))
            offset = offset + self.spatial_att[level](diff)
            if i == 3:
                offset = self.lrelu(self.offset_conv3[level](offset))
            else:
                offset = self.lrelu(self.offset_conv3[level](torch.cat([offset, upsampled_offset], dim=1)))
            feat = self.dcn_pack[level](nbr_feat_l[i - 1], offset)
            if i < 3:
                feat = self.feat_conv[level](torch.cat([feat, upsampled_feat], dim=1))
            if i > 1:
                feat = self.lrelu(feat)
                upsampled_offset = self.upsample(offset) * 2
                upsampled_feat = self.upsample(feat)
        return feat


class CasSGAB(nn.Module):
    def __init__(self, in_channels, deformable_groups=8):
        super(CasSGAB, self).__init__()
        self.offset_conv1 = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1)
        self.offset_conv2 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.offset_conv3 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.diff_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.spatial_att = SpatialAttention(in_channels)
        self.dcnpack = DCNv2Pack(in_channels, in_channels, 3, padding=1, deformable_groups=deformable_groups)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, coarse_feat, reference_feat):
        offset = self.lrelu(self.offset_conv1(torch.cat([coarse_feat, reference_feat], dim=1)))
        offset = self.lrelu(self.offset_conv2(offset))
        diff = reference_feat - coarse_feat
        diff = self.lrelu(self.diff_conv(diff))
        offset = offset + self.spatial_att(diff)
        offset = self.lrelu(self.offset_conv3(offset))
        feat = self.lrelu(self.dcnpack(reference_feat, offset))
        return feat


@ARCH_REGISTRY.register()
class AVDNet(nn.Module):
    def __init__(self, num_feat=64):
        super(AVDNet, self).__init__()
        # extract
        self.feat_extract = nn.Conv2d(3, num_feat // 4, 3, 1, 1)

        # demoireing
        self.down1 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        self.demoire1_1 = ABB(num_feat, num_feat // 2)
        self.down2 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        self.demoire2_1 = ABB(num_feat, num_feat // 2)
        self.down3 = nn.Conv2d(num_feat, num_feat // 4, 3, 1, 1)
        self.demoire3_1 = ABB(num_feat, num_feat // 2)
        self.concat3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.demoire3_2 = ABB(num_feat, num_feat // 2)
        self.concat2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.demoire2_2 = ABB(num_feat, num_feat // 2)
        self.concat1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.demoires1_2 = ABB(num_feat, num_feat // 2)

        # alignment
        self.coarse_align = SGAB(num_feat, deformable_groups=8)
        self.fusion = nn.Conv2d(num_feat * 3, num_feat, 3, 1, 1)
        self.fine_align = CasSGAB(num_feat, deformable_groups=8)

        # refinement
        self.up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.refine1 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.up2 = nn.Conv2d(num_feat // 4, num_feat, 3, 1, 1)
        self.refine2 = nn.Conv2d(num_feat // 4, num_feat // 4, 3, 1, 1)
        self.out_conv = nn.Conv2d(num_feat // 4, 3, 3, 1, 1)

        # others
        self.pus = nn.PixelUnshuffle(2)
        self.ps = nn.PixelShuffle(2)
        self.bilinear_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        b, t, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        feat_0 = self.pus(self.feat_extract(x))
        
        # demoireing
        feat_l1 = self.pus(self.down1(feat_0))
        feat_l1= self.demoire1_1(feat_l1)
        feat_l2 = self.pus(self.down2(feat_l1))
        feat_l2 = self.demoire2_1(feat_l2)
        feat_l3 = self.pus(self.down3(feat_l2))
        feat_l3 = self.demoire3_1(feat_l3)
        feat_l3 = self.concat3(feat_l3)
        up_l2 = self.bilinear_up(feat_l3)
        feat_l3 = self.demoire3_2(feat_l3)
        feat_l2 = self.leaky_relu(self.concat2(torch.cat([feat_l2, up_l2], dim=1)))
        up_l1 = self.bilinear_up(feat_l2)
        feat_l2 = self.demoire2_2(feat_l2)
        feat_l1 = self.leaky_relu(self.concat1(torch.cat([feat_l1, up_l1], dim=1)))
        feat_l1 = self.demoires1_2(feat_l1)

        # alignment
        feat_l1 = feat_l1.view(b, t, -1, h // 4, w // 4)
        feat_l2 = feat_l2.view(b, t, -1, h // 8, w // 8)
        feat_l3 = feat_l3.view(b, t, -1, h // 16, w // 16)
        ref_feat_l = [feat_l1[:, 1, :, :, :].clone(), feat_l2[:, 1, :, :, :].clone(), feat_l3[:, 1, :, :, :].clone()]
        aligned_feat = []
        for i in range(t):
            nbr_feat_l = [
                feat_l1[:, i, :, :, :].clone(), feat_l2[:, i, :, :, :].clone(),
                feat_l3[:, i, :, :, :].clone()
            ]
            aligned_feat.append(self.coarse_align(nbr_feat_l, ref_feat_l))
        aligned_feat = torch.stack(aligned_feat, dim=1)
        aligned_feat = aligned_feat.view(b, -1, h // 4, w // 4)
        aligned_feat = self.leaky_relu(self.fusion(aligned_feat))
        aligned_feat = self.fine_align(feat_l1[:, i, :, :, :].clone(), aligned_feat)

        # refinement
        aligned_feat = self.ps(self.up1(aligned_feat))
        aligned_feat = self.leaky_relu(self.refine1(aligned_feat))
        aligned_feat = self.ps(self.up2(aligned_feat))
        aligned_feat = self.leaky_relu(self.refine2(aligned_feat))
        out = self.out_conv(aligned_feat)
        return out