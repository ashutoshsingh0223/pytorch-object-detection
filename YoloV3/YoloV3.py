from typing import Union, Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np

from YoloV3.utils.parse_yolo_configs import get_layers_from_blocks, parse_yolo_config
from YoloV3.utils import predict_transform
from YoloV3.layers import DetectionLayer


class YoloV3(nn.Module):

    def __init__(self, config_path: Union['Path', str]):
        super(YoloV3, self).__init__()

        self.blocks = parse_yolo_config(config_path)
        self.net_info, self.module_list = get_layers_from_blocks(blocks=self.blocks)

        self.yolo_layers = [layer[0]
                            for layer in self.module_list if isinstance(layer[0], DetectionLayer)]

    def forward(self, x, target: Optional[Dict[str, Any]] = None, device: str = torch.device('cpu')):
        modules = self.blocks[1:]
        # Catch outputs for route layer
        outputs = {}

        write = False
        for i, module in enumerate(modules):
            type_ = module['type']

            if type_ == 'convolutional' or type_ == 'upsample':
                x = self.module_list[i](x)

            elif type_ == 'route':
                _layers = module['layers']

                if isinstance(_layers, int):
                    end = None
                    start = _layers
                else:
                    _layers = _layers.split(',')
                    start = int(_layers[0].strip())
                    end = int(_layers[1].strip())

                if start > 0:
                    start = start - i
                if end is not None:
                    end = end - i if end > 0 else end

                if end:
                    x = torch.cat((outputs[i + start], outputs[i + end]), dim=1)
                else:
                    x = outputs[i + start]
            elif type_ == 'shortcut':
                from_ = int(module['from'])

                x = outputs[i - 1] + outputs[i + from_]

            elif type_ == 'yolo':
                anchors = self.module_list[i][0].anchors
                # Get the input dimensions
                inp_dim = int(self.net_info["height"])

                # Get the number of classes
                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, device)

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile: Union['Path', str]):
        # Open the weights file
        fp = open(weightfile, "rb")

        # The first 5 values are header information
        # 1. Major version number
        # 2. Minor Version Number
        # 3. Subversion number
        # 4,5. Images seen by the network (during training)
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]

        weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']

            if module_type == 'convolutional':
                try:
                    batch_normalize = self.blocks[i+1]['batch_normalize']
                except KeyError:
                    batch_normalize = False
                conv = self.module_list[i].block[0]

                if batch_normalize:
                    # Get the number of weights of Batch Norm Layer
                    bn = self.module_list[i].block[1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases

                    # Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copy the data to model
                    self.module_list[i].block[1].bias.data.copy_(bn_biases)
                    self.module_list[i].block[1].weight.data.copy_(bn_weights)
                    self.module_list[i].block[1].running_mean.copy_(bn_running_mean)
                    self.module_list[i].block[1].running_var.copy_(bn_running_var)

                    # No conv bias if batch norm for YOLOV3
                else:
                    # Number of biases
                    num_biases = conv.bias.numel()

                    # Load the weights
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    self.module_list[i].block[0].bias.data.copy_(conv_biases)

                # Load conv weights
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                self.module_list[i].block[0].weight.data.copy_(conv_weights)
