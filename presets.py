from typing import Tuple, Optional, Dict
from PIL import Image

from torch import Tensor

import transforms as T


class DetectionPresetTrain():
    def __init__(self, *, data_augmentation, hflip_prob=0.5, mean=(123.0, 117.0, 104.0)):
        pass



class DetectionPresetTrainResize():
    def __init__(self, *, size: Tuple[int, int],
                 padding_color: Tuple[int, int, int] = (128, 128, 128), hflip_prob: float = 0.5,
                 mean: Tuple[float, float, float] = (123.0, 117.0, 104.0), data_augmentation: str = 'default'):

        if data_augmentation == 'default':
            self.transforms = T.Compose(
                [
                    T.ResizeImageAspectRatioPreserve(size=size, padding_color=padding_color),
                    T.PILToTensor()
                ]
            )

        def __call__(self, img: Image, target: Optional[Dict[str, 'Tensor']] = None):
            return self.transforms(img, target)


class DetectionPresetEval:
    def __init__(self):
        self.transforms = T.Compose(
            [
                T.PILToTensor()
            ]
        )
    def __call__(self, img: 'Image', target: Optional[Dict[str, 'Tensor']] = None):
        return self.transforms(img, target)


class DetectionPresetEvalResize:
    def __init__(self, size: Tuple[int, int], padding_color: Tuple[int, int, int] = (128, 128, 128)):
        self.transforms = T.Compose(
            [
                T.ResizeImageAspectRatioPreserve(size=size, padding_color=padding_color),
                T.PILToTensor()
            ]
        )
    def __call__(self, img: 'Image', target: Optional[Dict[str, 'Tensor']] = None):
        return self.transforms(img, target)
