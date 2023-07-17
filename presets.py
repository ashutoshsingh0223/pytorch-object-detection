from typing import Tuple, Optional, Dict, List, Callable
from PIL import Image

from torch import Tensor

import transforms as T


class Preset:
    def __init__(self, transforms):
        self.transforms = T.Compose(transforms)

    def __call__(self, img: Image.Image, target: Optional[Dict[str, Tensor]] = None):
        return self.transforms(img, target)


class Train(Preset):
    def __init__(
        self,
        *,
        size: int,
        padding_color: Tuple[int, int, int] = (128, 128, 128),
        hflip_prob: float = 0.5,
        mean: Tuple[float, float, float] = (123.0, 117.0, 104.0),
        data_augmentation: str = "coco_yolov3",
        transforms: Optional[List[Callable]] = None
    ):
        if data_augmentation == "coco_yolov3":
            transforms = [
                T.ResizeImageAspectRatioPreserve(size=416, padding_color=padding_color),
                T.RelativeBoxes(),
            ]

        super(Train, self).__init__(transforms=transforms)


class Eval(Preset):
    def __init__(
        self,
        size: Tuple[int, int],
        padding_color: Tuple[int, int, int] = (128, 128, 128),
        data_augmentation: str = "coco_yolo3",
        transforms: Optional[List[Callable]] = None,
    ):
        if data_augmentation == "coco_yolo3":
            transforms = transforms or [
                T.ResizeImageAspectRatioPreserve(
                    size=size, padding_color=padding_color
                ),
                T.RelativeBoxes(),
            ]

        super(Eval, self).__init__(transforms=transforms)
