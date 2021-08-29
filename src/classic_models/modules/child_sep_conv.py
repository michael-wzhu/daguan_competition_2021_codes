# -*- coding: utf-8 -*-
"""
@File: child_sep_conv.py
@Copyright: 2019 Michael Zhu
@License：the Apache License, Version 2.0
@Author：Michael Zhu
@version：
@Date：
@Desc: 
"""

import torch.nn as nn
import torch.nn.functional as F


class ChildSepConv(nn.Module):
    """
    Depthwise Separable Conv
    """

    def __init__(self, in_ch,
                 out_ch,
                 kernel_size,):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size

        self.depthwise_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(in_ch),
            kernel_size=int(kernel_size),
            groups=int(in_ch),
            padding=int(kernel_size // 2),
            bias=False
        )

        self.pointwise_conv = nn.Conv1d(
            in_channels=int(in_ch),
            out_channels=int(out_ch),
            kernel_size=1,
            padding=0,
            bias=False
        )

        self.op = nn.Sequential(
            self.depthwise_conv,
            nn.ReLU(inplace=False),
            self.pointwise_conv,
        )

    def forward(self, x, mask=None):
        x = x.transpose(1, 2)
        # print("x: ", x.size())
        x_conv = self.op(x)
        # print("x: ", x.size())
        x_conv = x_conv.transpose(1, 2)
        # print("x_conv: ", x_conv.size())

        if self.kernel_size % 2 == 0:
            x_conv = x_conv[:, :, :-1]

        return x_conv



if __name__ == "__main__":
    pass
    
