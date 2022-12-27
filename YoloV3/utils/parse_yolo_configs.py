from typing import Union, List, Dict, Any, Tuple
from pathlib import Path
from collections import OrderedDict

import torch.nn as nn

from blocks.conv_blocks import Conv2DBlock
from YoloV3.layers import EmptyLayer, DetectionLayer


activation_map_from_yolo = {
    'linear': 'linear',
    'leaky': 'leaky_relu',
}


def parse_yolo_config(path: Union['Path', str]) -> List[Dict[str, Any]]:
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
            block['type'] = line[1: -1].strip()
        else:
            key, value = line.split('=')
            key, value = key.strip(), value.strip()
            try:
                value = int(value)
            except ValueError:
                pass

            block[key.strip()] = value

    blocks.append(block)
    return blocks


def get_layers_from_blocks(blocks: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], 'nn.ModuleList']:
    module_list = nn.ModuleList()
    in_channels: int = 3
    out_channels_list: List[int] = []

    for index, block in enumerate(blocks[1:]):
        type_ = block['type']
        module_block = nn.Sequential()
        if type_ == 'convolutional':
            try:
                batch_norm = int(block.get('batch_normalize'))
            except:
                batch_norm = 0

            batch_norm = bool(batch_norm)

            activation = activation_map_from_yolo[block['activation']]
            out_channels = int(block['filters'])
            kernel_size = int(block['size'])
            stride = int(block['stride'])

            try:
                padding = int(block['pad'])
                padding = (kernel_size - 1) // 2 if padding else 0
            except:
                padding = 0

            # Conv bias is set to false if batch norm present
            module_block = Conv2DBlock(in_channels, out_channels, kernel_size,
                                       stride, padding, batch_norm, activation, conv_bias=not batch_norm)
        elif type_ == 'shortcut':
            module_block.add_module(f'shortcut_{index}', EmptyLayer())

        elif type_ == 'upsample':
            stride = block['stride']
            module_block.add_module(f'upsample_{index}', nn.Upsample(scale_factor=stride, mode='bilinear'))

        elif type_ == 'route':
            _layers = block['layers']
            module_block.add_module(f'route_{index}', EmptyLayer())

            if isinstance(_layers, str):
                _layers = _layers.split(',')
                start = int(_layers[0].strip())
                end = int(_layers[1].strip())
            else:
                start = _layers
                end = 0

            # Since route start layer index is always less than route layer index, the start number is always negative
            # in case instead if relative negative index like -1(for previous layer) is replaced by actual layer index
            # like 10(10th layer) ; like it's done for end layer this handling will help the downstream code to be
            # consistent.
            if start > 0:
                start = start - index
            if end > 0:
                end = end - index

            #  If end is given out channels for this index will be concatenation of out channels from start layer and
            # end layer.
            if end < 0:
                out_channels = out_channels_list[start + index] + out_channels_list[end + index]
            else:
                out_channels = out_channels_list[start + index]

        elif type_ == 'yolo':
            masks = block['mask'].split(',')
            masks = [int(m.strip()) for m in masks]
            anchors = [int(a.strip()) for a in block['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in masks]
            module_block.add_module(f'detection_{index}', DetectionLayer(anchors=anchors))

        in_channels = out_channels
        out_channels_list.append(out_channels)
        module_list.append(module_block)

    return blocks[0], module_list
