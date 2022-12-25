from typing import Union, Any, List
from pathlib import Path

from torchvision.datasets import CocoDetection

import transforms as T
from base_constants import TRAIN, VALIDATION, TEST, COCO_BOX


def get_coco(image_dir: Union['Path', str], annotations_file: Union['Path', str], transforms: List[Any],
             mode: str = TRAIN, iou_types: List[str] = [COCO_BOX]):
    transforms = T.Compose(transforms)
    dataset = CocoDetection(root=image_dir, annFile=annotations_file, transforms=transforms)

    if mode == TRAIN:
        pass
        # dataset = _coco_remove_images_without_annotations(dataset, iou_types=iou_types)

    return dataset
