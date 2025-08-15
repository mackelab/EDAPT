"""ShallowConvNet implementation adapted from Schirrmeister et al. 2017."""

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from .utils import _glorot_weight_zero_bias


class Square(nn.Module):
    """Square activation function."""
    
    def forward(self, x):
        return torch.square(x)


class Log(nn.Module):
    """Log activation with numerical stability."""
    
    def forward(self, x):
        return torch.log(torch.clamp(x, min=1e-6))


class ShallowConvNet(nn.Module):
    """ShallowConvNet from Schirrmeister et al. 2017."""

    def __init__(self, n_chans, n_outputs, n_times, cnn_temporal_kernels=40, 
                 cnn_temporal_kernelsize=(25, 1), dilation1_time=1, cnn_spatial_kernels=40,
                 cnn_poolsize=(75, 1), cnn_poolstride=(15, 1), cnn_pool_type="avg", dropout=0.5):
        super().__init__()
        self.n_chans = n_chans
        self.n_outputs = n_outputs
        self.n_times = n_times
        
        if cnn_spatial_kernels != cnn_temporal_kernels:
            cnn_spatial_kernels = cnn_temporal_kernels

        # Input adapter: (B, C, T) -> (B, T, C, 1) -> (B, 1, T, C)
        self.input_adapter = Rearrange("b c t -> b t c 1")

        # Convolutional layers
        self.conv_module = nn.Sequential(
            nn.Conv2d(1, cnn_temporal_kernels, cnn_temporal_kernelsize, 
                     padding="valid", bias=False, dilation=(dilation1_time, 1)),
            nn.Conv2d(cnn_temporal_kernels, cnn_spatial_kernels, (1, n_chans), 
                     padding="valid", bias=False),
            nn.BatchNorm2d(cnn_spatial_kernels, momentum=0.1, affine=True),
            Square(),
            nn.AvgPool2d(cnn_poolsize, stride=cnn_poolstride) if cnn_pool_type == "avg" 
                else nn.MaxPool2d(cnn_poolsize, stride=cnn_poolstride),
            Log(),
            nn.Dropout(p=dropout)
        )

        # Calculate dense layer input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, n_chans, n_times)
            adapted_dummy_input = self.input_adapter(dummy_input)
            permuted_dummy_input = adapted_dummy_input.permute(0, 3, 1, 2)
            dummy_output = self.conv_module(permuted_dummy_input)
            dense_input_size = self._num_flat_features(dummy_output)

        # Dense layers
        self.dense_module = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dense_input_size, n_outputs, bias=True)
        )

        _glorot_weight_zero_bias(self)

    def _num_flat_features(self, x):
        """Calculate number of flattened features."""
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Forward pass expecting (batch, channels, time) input."""
        x = self.input_adapter(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x