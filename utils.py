from typing import Tuple, Any, Dict

import torch.nn as nn


def get_activation_and_params(name) -> Tuple[Any, Dict[str, Any]]:
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
        'softmax': nn.Softmax
    }

    params_index = {
        'relu': {'inplace': True},
        'sigmoid': {},
        'relu6': {'inplace': True},
        'leaky_relu': {'negative_slope': 0.01, 'inplace': True},
        'softmax': {'dim': 1}
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


