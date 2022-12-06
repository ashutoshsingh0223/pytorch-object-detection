from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import cv2


def predict_transform(prediction: 'torch.Tensor', input_dim: int, anchors: List[Tuple[int, int]], num_classes: int,
                      cuda: bool = True):

    batch_size = prediction.shape[0]
    stride = input_dim // prediction.shape[2]
    grid_size = input_dim // stride

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Changing scale of anchor box from image size to feature-map of grid_size
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Sigmoid the centre x, y co-ords and objectness score
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add grid offsets to centre co-ordinates predictions

    grid = np.arange(grid_size)
    x, y = np.meshgrid(grid, grid)

    x = torch.FloatTensor(x).view(-1, 1)
    y = torch.FloatTensor(y).view(-1, 1)

    # Concat to form (x,y) offsets and repeat each offset as many times as num_anchors, add one more dimention to
    # handle directly adding to prediction tensor(for each pred in entire batch)
    offsets = torch.cat((x, y), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += offsets
    return prediction
