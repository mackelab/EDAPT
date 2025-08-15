"""DeepConvNet implementation from Schirrmeister et al. 2017."""

import math
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .utils import Conv2dWithConstraint, _glorot_weight_zero_bias


class DeepConvNet(nn.Module):
    """DeepConvNet from Schirrmeister et al. 2017."""

    def __init__(self, n_chans, n_outputs, n_times, n_filters_time=25, filter_time_length=10,
                 dilation1=1, n_filters_spat=25, n_filters_2=50, filter_length_2=10,
                 n_filters_3=100, filter_length_3=10, n_filters_4=200, filter_length_4=10,
                 pool_time_length=3, pool_time_stride=3, drop_prob=0.5, final_conv_length="auto", **kwargs):
        super().__init__()

        if final_conv_length == "auto":
            self.final_conv_length_calculated = self._get_final_conv_length(
                n_times, filter_time_length, dilation1, filter_length_2,
                filter_length_3, filter_length_4, pool_time_length, pool_time_stride)
        else:
            self.final_conv_length_calculated = final_conv_length

        # Block 1: Temporal + Spatial convolution
        padding_temporal = (dilation1 * (filter_time_length - 1)) // 2
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, n_filters_time, (filter_time_length, 1), bias=False,
                     padding=(padding_temporal, 0), dilation=(dilation1, 1)),
            Conv2dWithConstraint(n_filters_time, n_filters_spat, (1, n_chans), 
                               max_norm=2.0, bias=False),
            nn.BatchNorm2d(n_filters_spat, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((pool_time_length, 1), stride=(pool_time_stride, 1)),
            nn.Dropout(p=drop_prob)
        )

        # Subsequent blocks
        self.conv_block2 = self._create_conv_block(n_filters_spat, n_filters_2, filter_length_2, drop_prob)
        self.conv_block3 = self._create_conv_block(n_filters_2, n_filters_3, filter_length_3, drop_prob)
        self.conv_block4 = self._create_conv_block(n_filters_3, n_filters_4, filter_length_4, drop_prob)

        # Final classifier
        self.final_conv = nn.Conv2d(n_filters_4, n_outputs, (self.final_conv_length_calculated, 1), bias=True)

        # Input adapters
        self.to_b_1_c_t = Rearrange("b c t -> b 1 c t")

        _glorot_weight_zero_bias(self)

    def _create_conv_block(self, in_filters, out_filters, kernel_length, dropout):
        """Create standard convolution block."""
        return nn.Sequential(
            nn.Conv2d(in_filters, out_filters, (kernel_length, 1), bias=False,
                     padding=((kernel_length - 1) // 2, 0)),
            nn.BatchNorm2d(out_filters, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d((3, 1), stride=(3, 1)),
            nn.Dropout(p=dropout)
        )

    def _get_final_conv_length(self, n_times, f1, d1, f2, f3, f4, p1_k, p1_s):
        """Calculate final convolution length after all layers."""
        def conv_len(l_in, kernel, stride=1, dilation=1, padding=0):
            return math.floor(((l_in + 2 * padding - dilation * (kernel - 1) - 1) / stride) + 1)

        def pool_len(l_in, kernel, stride, padding=0):
            return math.floor(((l_in + 2 * padding - kernel) / stride) + 1)

        len_ = n_times
        # Block 1
        padding1 = (d1 * (f1 - 1)) // 2
        len_ = conv_len(len_, f1, stride=1, dilation=d1, padding=padding1)
        len_ = pool_len(len_, p1_k, p1_s)
        
        # Blocks 2-4 with fixed pooling
        for f in [f2, f3, f4]:
            padding = (f - 1) // 2
            len_ = conv_len(len_, f, padding=padding)
            len_ = pool_len(len_, 3, 3)
        
        return int(len_)

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.to_b_1_c_t(x)
        x = x.permute(0, 1, 3, 2)  # -> (B, 1, T, C)
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = self.final_conv(x)
        logits = x.squeeze(-1).squeeze(-1)
        return logits