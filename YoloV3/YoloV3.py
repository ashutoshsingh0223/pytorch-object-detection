from typing import Union, Optional, Dict, Any, List
from pathlib import Path

import torch
import torch.nn as nn

import numpy as np

from YoloV3.utils.parse_yolo_configs import (
    parse_yolo_config,
    get_layers_from_blocks,
)
from YoloV3 import loss
from YoloV3.layers import DetectionLayer
from YoloV3.utils.transform_utils import predict_transform


class YoloV3(nn.Module):
    def __init__(self, config_path: Union[Path, str], hyperparams: Dict[str, Any]):
        super(YoloV3, self).__init__()

        self.blocks = parse_yolo_config(config_path)
        self.net_info, self.module_list = get_layers_from_blocks(blocks=self.blocks)

        self.hyperparams = hyperparams

        self.yolo_layers = [
            layer[0]
            for layer in self.module_list
            if isinstance(layer[0], DetectionLayer)
        ]

    def calculate_loss(
        self, predictions: List[torch.Tensor], targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        boxes = torch.cat([target["boxes"] for target in targets], 0)
        losses = loss.calculate_loss(predictions, boxes, self, self.hyperparams)

        return losses

    def forward(
        self, x: torch.Tensor, targets: Optional[Dict[str, torch.Tensor]] = None
    ):
        modules = self.blocks[1:]
        # Catch outputs for route layer
        outputs = {}
        yolo_detections = []

        if self.training and targets is None:
            torch._assert(False, "`targets` cannot be none while training")

        write = False
        for i, module in enumerate(modules):
            type_ = module["type"]

            if type_ == "convolutional" or type_ == "upsample":
                x = self.module_list[i](x)

            elif type_ == "route":
                _layers = module["layers"]

                if isinstance(_layers, int):
                    end = None
                    start = _layers
                else:
                    _layers = _layers.split(",")
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

            elif type_ == "shortcut":
                from_ = int(module["from"])

                x = outputs[i - 1] + outputs[i + from_]

            elif type_ == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                # Get the number of classes
                num_classes = int(module["classes"])

                # x = self.module_list[i][0](x, inp_dim)
                # yolo_detections.append(x)
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, device=x.device)

                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        if self.training:
            losses = self.calculate_loss(yolo_detections, targets)
            return losses

        return detections

    def load_weights(self, weightfile: Union[Path, str]):
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
            module_type = self.blocks[i + 1]["type"]

            if module_type == "convolutional":
                try:
                    batch_normalize = self.blocks[i + 1]["batch_normalize"]
                except KeyError:
                    batch_normalize = False

                conv = self.module_list[i][0]

                if batch_normalize:
                    # Get the number of weights of Batch Norm Layer
                    bn = self.module_list[i][1]
                    num_bn_biases = bn.bias.numel()

                    bn_biases = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr : ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(
                        weights[ptr : ptr + num_bn_biases]
                    )
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
                    conv_biases = torch.from_numpy(weights[ptr : ptr + num_biases])
                    ptr = ptr + num_biases

                    # reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Finally copy the data
                    conv.bias.data.copy_(conv_biases)

                # Load conv weights
                num_weights = conv.weight.numel()

                # Do the same as above for weights
                conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
