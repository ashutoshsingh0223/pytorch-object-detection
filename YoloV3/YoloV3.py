from typing import Union
from pathlib import Path

import torch
import torch.nn as nn


from YoloV3.utils.parse_yolo_configs import get_layers_from_blocks, parse_yolo_config
from YoloV3.utils import predict_transform


class YoloV3(nn.Module):

    def __int__(self, config_path: Union['Path', str]):
        super(YoloV3, self).__int__()

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
