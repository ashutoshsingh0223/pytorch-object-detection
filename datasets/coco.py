from typing import Union, Any, List, Optional, Callable, Tuple
from pathlib import Path
import warnings

from PIL import Image
import cv2
import random

from torchvision.datasets import VisionDataset
import torch.nn.functional as F
import torch

import transforms as T
from presets import Preset
from base_constants import TRAIN, VALIDATION, TEST, COCO_BOX


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="bilinear").squeeze(0)
    return image


class CocoDetection(VisionDataset):
    """
    Copied and modified version of torchvision.datasets.CocoDetection

    `MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        images_dir (string): Root directory where images are downloaded to.
        annotations_file (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        images_dir: str,
        annotations_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Preset] = None,
        size_range: Optional[Tuple[int, int]] = None,
        batch_count_for_multiscale: Optional[int] = None,
        yolo: bool = False,
    ) -> None:
        super().__init__(images_dir, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.images_dir: "Path" = Path(images_dir)

        self.images_dim = []
        self.batch_count = 0

        self.yolo = yolo
        self.size_range = size_range
        self.batch_count_for_multiscale = batch_count_for_multiscale

        if (self.yolo and self.transforms is not None) and not isinstance(
            self.transforms.transforms.transforms[-1], T.RelativeBoxes
        ):
            torch._assert(
                False,
                "When yolo is set for coco dataset, the last transform should be tranforms.RelativeBoxes",
            )

        if self.size_range is not None:
            warnings.warn(
                "Using multiscale training. targets will not scaled and will be in original form, either absolute or relative if yolo is set"
            )

        if annotations_file:
            self.coco = COCO(annotations_file)
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.paths = None
        else:
            self.coco = None
            self.ids = None
            # If annotations file is not specified take all the paths from image_dir
            self.paths = [x for x in self.images_dir.glob("**/*") if x.is_file()]

    def _load_image(self, _id: int) -> Image.Image:
        path = self.coco.loadImgs(_id)[0]["file_name"]
        return Image.open(str(Path(self.root) / path))

    def _load_target(self, _id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(_id))

    def _load_images_from_dir(self, index: int) -> Image.Image:
        return Image.open(str(self.paths[index]))
        # return Image.open(str(self.paths[index]))

    def get_img_path(self, index: int) -> Union["Path", str]:
        if self.coco:
            _id = self.ids[index]
            path = self.coco.loadImgs(_id)[0]["file_name"]
            return Path(self.root) / path
        else:
            return self.paths[index]

    def get_img_dim(self, index: int) -> Tuple[int, int]:
        if not self.images_dim:
            if self.coco:
                self.images_dim = [
                    Image.open(
                        Path(self.root) / self.coco.loadImgs(_id)[0]["file_name"]
                    ).size
                    for _id in self.ids
                ]
            else:
                self.images_dim = [Image.open(p).size for p in self.paths]

        return self.images_dim[index]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.coco:
            _id = self.ids[index]
            image = self._load_image(_id)
            target = self._load_target(_id)

        else:
            image = self._load_images_from_dir(index)
            target = {}

        image, target = T.ConvertCocoPolysToMask()(image, target)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        if self.coco:
            return len(self.ids)
        else:
            return len(self.paths)

    def collate_fn(self, batch: List[Tuple[Any, ...]]) -> Tuple[Any, ...]:
        images, targets = tuple(zip(*batch))

        self.batch_count += 1

        if (
            self.size_range is not None
            and self.batch_count % self.batch_count_for_multiscale
        ):
            size = random.choice(range(self.size_range[0], self.size_range[1] + 1, 32))
            images = torch.stack([resize(img, size) for img in images])

        if self.yolo:
            for i, target in enumerate(targets):
                boxes = targets["boxes"]
                _boxes = torch.zeros(
                    (boxes.shape[0], boxes.shape[1] + 2),
                    dtype=boxes.dtype,
                    device=boxes.device,
                )
                _boxes[:, 0] = i
                _boxes[:, 1] = target["labels"]
                _boxes[:, 2:] = boxes
                target["boxes"] = _boxes
        return images, targets


# from yolo_detector import *
# from argparse import Namespace
#
# args = Namespace()
# args.data_augmentation = 'default'
# args.resolution = 416
#
# val_dataset = get_dataset('coco', datapath=('/home/AD.IGD.FRAUNHOFER.DE/sashutosh/pytorch-object-detection/test_images'), mode=VALIDATION, transforms=get_transform(False, args))
# val_loader = DataLoader(dataset=val_dataset, num_workers=2, shuffle=False, batch_size=1)
