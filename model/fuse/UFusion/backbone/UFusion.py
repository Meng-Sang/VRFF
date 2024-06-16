#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Time     :2022/7/119:32
@Author   :dan wu
@FileName :gen.py
@Software :PyCharm
"""
import cv2
import numpy as np

# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
File       : Generator.py
Create on  ：2021/7/26 14:03

Author     ：yujing_rao
"""
import torch
from torch import nn
import kornia as k


def gaussian_blur2d(input):
    x_gau = k.filters.gaussian_blur2d(input, (13, 13), (1, 1))
    return x_gau


# device ='cuda'
class DSC(nn.Module):
    def __init__(self, in_channels, out_channel, kernel_size=3, stride=1, bias=True):
        super(DSC, self).__init__()
        self.conv1_1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1,
                                                        groups=in_channels, bias=bias))
        self.conv3_3 = nn.utils.spectral_norm(
            nn.Conv2d(in_channels, out_channel, kernel_size=1, stride=stride, groups=1, bias=bias))

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv3_3(x)
        return x


def get_sn_conv(in_ch, out_ch, kernel_size=3, stride=1, bias=True, fast=True):
    if fast:
        Conv2d = DSC(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias)
    else:
        Conv2d = nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias))
    return Conv2d


class UFusion(nn.Module):
    def __init__(self, device):
        super(UFusion, self).__init__()
        self.device = device
        self.pad = nn.ReflectionPad2d(1)
        self.leaky_relu = nn.LeakyReLU(0.2)

        self.sn_conv1 = get_sn_conv(1, 8, kernel_size=3, stride=1, bias=True)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=8)

        self.sn_conv2 = get_sn_conv(8, 16, kernel_size=3, stride=1, bias=True)
        self.gn2 = nn.GroupNorm(num_groups=2, num_channels=16)

        self.sn_conv2_2 = get_sn_conv(16, 16, kernel_size=3, stride=1, bias=True)
        self.gn2_2 = nn.GroupNorm(num_groups=2, num_channels=16)

        self.sn_conv3 = get_sn_conv(16, 32, kernel_size=3, stride=1, bias=True)
        self.gn3 = nn.GroupNorm(num_groups=2, num_channels=32)

        self.sn_conv3_2 = get_sn_conv(32, 32, kernel_size=3, stride=1, bias=True)
        self.gn3_2 = nn.GroupNorm(num_groups=2, num_channels=32)
        # self.gn3 = nn.GroupNorm(num_groups=2,num_channels=64)

        self.sn_conv4 = get_sn_conv(32, 64, kernel_size=3, stride=1, bias=True)
        self.gn4 = nn.GroupNorm(num_groups=2, num_channels=64)
        self.sn_conv4_2 = get_sn_conv(64, 64, kernel_size=3, stride=1, bias=True)
        self.gn4_2 = nn.GroupNorm(num_groups=2, num_channels=64)

        self.sn_conv5 = get_sn_conv(128, 32, kernel_size=3, stride=1, bias=True)
        self.gn5 = nn.GroupNorm(num_groups=2, num_channels=32)

        self.sn_conv6 = get_sn_conv(96, 16, kernel_size=3, stride=1, bias=True)
        self.gn6 = nn.GroupNorm(num_groups=2, num_channels=16)
        self.sn_conv7 = get_sn_conv(48, 8, kernel_size=3, stride=1, bias=True)
        self.sn_conv8 = get_sn_conv(24, 1, kernel_size=1, stride=1, bias=True)

        for name, p in self.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(p)
            elif 'bn' in name:
                nn.init.trunc_normal_(p, mean=1, std=1e-3)
            else:
                nn.init.trunc_normal_(p, std=1e-3)

    def pre_process(self, ir_imgs, vi_imgs):
        padding = 1
        sub_ir_sequence = []
        sub_vi_sequence = []
        input_ir = (ir_imgs - 127.5) / 127.5

        input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
        w2, h2 = input_ir.shape
        input_ir = input_ir.reshape([w2, h2, 1])
        input_vi = (vi_imgs - 127.5) / 127.5

        input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
        w4, h4 = input_vi.shape
        input_vi = input_vi.reshape([w4, h4, 1])

        sub_ir_sequence.append(input_ir)
        sub_vi_sequence.append(input_vi)

        train_data_ir = np.asarray(sub_ir_sequence)
        train_data_vi = np.asarray(sub_vi_sequence)

        train_data_ir = train_data_ir.transpose([0, 3, 1, 2])
        train_data_vi = train_data_vi.transpose([0, 3, 1, 2])
        train_data_ir = torch.tensor(train_data_ir).float().to(self.device)
        train_data_vi = torch.tensor(train_data_vi).float().to(self.device)

        return train_data_ir, train_data_vi

    @staticmethod
    def post_process(result):
        result = np.squeeze(result.detach().cpu().numpy() * 127.5 + 127.5).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(result)
        return result

    def forward(self, ir_imgs, vi_imgs):
        ir_imgs, vi_imgs = self.pre_process(ir_imgs, vi_imgs)
        ir_imgs = gaussian_blur2d(ir_imgs)
        vi_imgs = gaussian_blur2d(vi_imgs)
        x1v = self.leaky_relu(self.gn1(self.sn_conv1(vi_imgs)))  # 8
        x1i = self.leaky_relu(self.gn1(self.sn_conv1(ir_imgs)))
        x2v = self.pad(self.leaky_relu(self.gn2(self.sn_conv2(x1v))))  # 16
        x2i = self.pad(self.leaky_relu(self.gn2(self.sn_conv2(x1i))))
        xv = self.pad(self.leaky_relu(self.gn2_2(self.sn_conv2_2(x2v))))
        xi = self.pad(self.leaky_relu(self.gn2_2(self.sn_conv2_2(x2i))))
        r_x2v = x2v + xv
        r_x2i = x2i + xi
        x3v = self.pad(self.leaky_relu(self.gn3(self.sn_conv3(r_x2v))))  # 32
        x3i = self.pad(self.leaky_relu(self.gn3(self.sn_conv3(r_x2i))))
        xv = self.pad(self.leaky_relu(self.gn3_2(self.sn_conv3_2(x3v))))
        xi = self.pad(self.leaky_relu(self.gn3_2(self.sn_conv3_2(x3i))))
        r_x3v = xv + x3v
        r_x3i = xi + x3i
        x4v = self.pad(self.leaky_relu(self.gn4(self.sn_conv4(r_x3v))))  # 64
        x4i = self.pad(self.leaky_relu(self.gn4(self.sn_conv4(r_x3i))))
        xv = self.pad(self.leaky_relu(self.gn4_2(self.sn_conv4_2(x4v))))
        xi = self.pad(self.leaky_relu(self.gn4_2(self.sn_conv4_2(x4i))))
        r_x4v = xv + x4v
        r_x4i = xi + x4i
        x4 = torch.cat([r_x4v, r_x4i], dim=1)  # 128
        x5 = self.pad(self.leaky_relu(self.gn5(self.sn_conv5(x4))))  # 32
        x5 = torch.cat([x3i, x3v, x5], dim=1)
        x6 = self.pad(self.leaky_relu(self.gn6(self.sn_conv6(x5))))  # 16
        x6 = torch.cat([x2i, x2v, x6], dim=1)
        x7 = self.pad(self.leaky_relu(self.sn_conv7(x6)))  # 8
        x7 = torch.cat([x1i, x1v, x7], dim=1)
        x8 = torch.tanh(self.sn_conv8(x7))
        return self.post_process(x8)
