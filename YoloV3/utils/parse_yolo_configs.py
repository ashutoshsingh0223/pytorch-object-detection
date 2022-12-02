from typing import Union, List, Dict, Any
from pathlib import Path
from collections import OrderedDict

import torch.nn as nn

from blocks.conv_blocks import Conv2DBlock


activation_map_from_yolo = {
    'linear': '',
    'leaky': 'leaky_relu'
}

def parse_yolo_config(path: Union['Path', str]):
    path = Path(path) if isinstance(path, str) else path
    cfg: List[str] = open(path, 'r').read().split('\n')

    lines: List[str] = [line.strip() for line in cfg if line.strip()]
    lines = [line for line in lines if line[0] != '#']

    blocks: List[Dict[str, Any]] = []
    block: Dict[str, Any] = OrderedDict()

    for line in lines:
        if line[0] == '[':
            if block:
                blocks.append(block)
            block = OrderedDict()
            block['type'] = line[1: -1]
        else:
            key, value = line.split('=')
            try:
                value = int(value)
            except ValueError:
                pass

            block[key] = value

    blocks.append(block)
    return blocks


def get_layers_from_blocks(blocks: List[Dict[str, Any]]):
    module_list = nn.ModuleList()
    in_channels = 3
    out_channels_list = []

    for block in blocks[1:]:
        type_ = block['type']

        if type_ == 'convolutional':
            try:
                batch_norm = int(block.get('batch_normalize'))
            except:
                batch_norm = 0

            activation = activation_map_from_yolo[block['activation']]
            out_channels = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            out_channels_list.append(out_channels)

            try:
                padding = int(block['pad'])
                padding = (kernel_size - 1) // 2 if padding else 0
            except:
                padding = 0

            module_block = Conv2DBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       padding=padding, stride=stride, activation=activation,
                                       batch_norm=bool(batch_norm))

            out_channels_list.append(out_channels)
            module_list.append(module_block)
        elif type_ == 'shortcut':
