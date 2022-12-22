from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


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

    anchors = torch.FloatTensor(anchors)
    if cuda:
        anchors = anchors.cuda()

    # Log space transform for height and width
    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    # Sigmoid on class scores. One bounding box may predict different classes, hence no softmax.
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))

    # Scale the bbox attributes to fit the size of image from the size of feature map.
    # Example: feature map size = 13x13 and image size = 416x416 then stride = 32 = 416 // 13
    prediction[:, :, :4] *= stride
    return prediction


def transform_detections(predictions: 'Tensor', confidence: float, num_classes: int, nms_threshold: float = 0.4):
    conf_mask = (predictions[:, :, 4] > confidence).float().unsqueeze(2)
    predictions = predictions * conf_mask

    # We have bbox in tuples like (centre_x, centre_y, width, height) but it's easier to calculate IoU
    # if we have upper left co-ordinate and lower right co-ordinate of the box
    box_corner = predictions.new(predictions.shape)
    box_corner[:, :, 0] = (predictions[:, :, 0] - predictions[:, :, 2] / 2)
    box_corner[:, :, 1] = (predictions[:, :, 1] - predictions[:, :, 3] / 2)
    box_corner[:, :, 2] = (predictions[:, :, 0] + predictions[:, :, 2] / 2)
    box_corner[:, :, 3] = (predictions[:, :, 1] + predictions[:, :, 3] / 2)
    predictions[:, :, :4] = box_corner[:, :, :4]

    for image_pred in predictions:
        # Out of 80 classes keep the class index with max score along with its score
        max_conf_score, max_conf_index = torch.max(image_pred[:, 5:5 + num_classes], 1, keep_dim=True)
        max_conf_score = max_conf_score.float()
        max_conf_index = max_conf_index.float()
        seq = (image_pred[:, :5], max_conf_score, max_conf_index)

        # Let's remove the bounding boxes we had set to zero by using 'confidence' or objectness score
        non_zero = torch.non_zero(image_pred[:, 4])
        # 7 becuase of 5 bbox attrs and 1 each for max class index and corresponding score

        try:
            image_pred = image_pred[non_zero.unsqueeze(1), :].view(-1, 7)
        except:
            continue

        # The above code does not raise exception for older torch versions where scalars are supported if image_pred
        # is empty

        if image_pred.shape[0] == 0:
            continue

        unique_classes = torch.unique(image_pred[:, -1])


