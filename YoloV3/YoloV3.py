from typing import Union
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np

from YoloV3.utils.parse_yolo_configs import get_layers_from_blocks, parse_yolo_config
from YoloV3.utils import predict_transform


class YoloV3(nn.Module):

    def __init__(self, config_path: Union['Path', str]):
        super(YoloV3, self).__init__()

        self.blocks = parse_yolo_config(config_path)
        self.net_info, self.module_list = get_layers_from_blocks(blocks=self.blocks)

    def forward(self, x, cuda: bool = True):
        modules = self.blocks[1:]
        # Catch outputs for route layer
        outputs = []

        write = False
        for i, module in enumerate(modules):
            type_ = module['type']

            if type_ == 'convolutional' or type_ == 'upsample':
                x = self.module_list[i](x)

            elif type_ == 'route':
                _layers = module['layers']

                if isinstance(_layers, int):
                    end = 0
                    start = _layers
                else:
                    _layers = _layers.split(',')
                    start = int(_layers[0].strip())
                    end = int(_layers[1].strip())

                if start > 0:
                    start = start - i
                if end > 0:
                    end = end - i

                if end < 0:
                    x = torch.cat((outputs[i + start] + outputs[i + end]), dim=1)
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
                x = predict_transform(x, inp_dim, anchors, num_classes, cuda)

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 0)

            outputs.append(x)

        return detections

    def load_weights(self, weightfile):
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
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

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
                    conv.bias.data.copy_(conv_biases)

                # Load conv weights
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr:ptr + num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
