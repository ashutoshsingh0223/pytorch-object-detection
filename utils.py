from typing import Tuple, Any, Dict, Optional

from PIL import Image
import numpy as np

import torch.nn as nn
from torch import Variable, from_numpy


def get_activation_and_params(name: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Get supported activation function with default params by name
    Args:
        name: name of activation function. any one from the index
    Returns:
        activation function class, dict of activation params
    """
    index = {
        'relu': nn.ReLU,
        'sigmoid': nn.Sigmoid,
        'relu6': nn.ReLU6,
        'leaky_relu': nn.LeakyReLU,
        'softmax': nn.Softmax,
        'linear': None,
    }

    params_index = {
        'relu': {'inplace': True},
        'sigmoid': {},
        'relu6': {'inplace': True},
        'leaky_relu': {'negative_slope': 0.01, 'inplace': True},
        'softmax': {'dim': 1},
        'linear': None
    }
    return index[name], params_index[name]


def get_pooling_from_params(params: Dict[str, Any]) -> 'nn.Module':
    index = {
        'max': nn.MaxPool2d,
        'avg': nn.AvgPool2d
    }
    type_ = params.pop('type')

    pooling_layer = index[type_](**params)
    return pooling_layer


def get_test_input(size: Tuple[int, int] = (416, 416), batch: bool = True):
    img = Image.open("dog-cycle-car.png")
    img = img.resize(size)          #Resize to the input dimension
    img = np.asarray(img)
    img_ = img.transpose((2, 0, 1))  # BGR -> RGB | H X W C -> C X H X W
    img_ = img_[np.newaxis, :, :, :]/255.0       #Add a channel at 0 (for batch) | Normalise
    img_ = from_numpy(img_).float()     #Convert to float
    img_ = Variable(img_)                     # Convert to Variable
    return img_


