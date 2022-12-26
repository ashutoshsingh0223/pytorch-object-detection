from typing import Dict, Optional, Any

import torch

from utils import get_activation_and_params, get_pooling_from_params


class Conv2DBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 batch_norm: bool, activation: str, conv_bias: bool = True, padding_mode: str = 'zeros',
                 activation_params: Optional[Dict[str, Any]] = None, pool_params: Optional[Dict[str, Any]] = None):
        """

        Args:
            in_channels:
            out_channels:
            kernel_size:
            stride:
            padding:
            batch_norm:
            activation:
            conv_bias:
            padding_mode:
            activation_params:
            pool_params:

        Returns:

        """
        super(Conv2DBlock, self).__init__()

        self.block = torch.nn.Sequential()

        self.block.add_module('conv', torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      padding_mode=padding_mode, bias=conv_bias))

        if batch_norm:
            self.block.add_module('batch_norm', torch.nn.BatchNorm2d(num_features=out_channels))

        activation, default_activation_params = get_activation_and_params(name=activation)
        if activation:
            activation_params = activation_params or default_activation_params
            self.block.add_module(f'activation', activation(**activation_params))

        if pool_params:
            self.block.add_module('pool', get_pooling_from_params(params=pool_params))

    def forward(self, x):
        x = self.block(x)
        return x
