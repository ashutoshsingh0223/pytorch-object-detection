from typing import Union, Any, List, Optional, Callable, Tuple
from pathlib import Path

from PIL import Image

from torchvision.datasets import VisionDataset

import transforms as T
from base_constants import TRAIN, VALIDATION, TEST, COCO_BOX


def get_coco(image_dir: Union['Path', str], annotations_file: Union['Path', str], transforms: Any,
             mode: str = TRAIN, iou_types: List[str] = [COCO_BOX]):
    t = []
    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)
    dataset = CocoDetection(images_dir=image_dir, annotations_file=annotations_file, transforms=transforms)

    if mode == TRAIN:
        pass
        # dataset = _coco_remove_images_without_annotations(dataset, iou_types=iou_types)

    return dataset


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
            transforms: Optional[Callable] = None,
    ) -> None:
        super().__init__(images_dir, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.images_dir: 'Path' = Path(images_dir)

        if annotations_file:
            self.coco = COCO(annotations_file)
            self.ids = list(sorted(self.coco.imgs.keys()))
            self.paths = None
        else:
            self.coco = None
            self.ids = None
            # If annotations file is not specified take all the paths from image_dir
            self.paths = [x for x in self.images_dir.glob('**/*') if x.is_file()]

    def _load_image(self, _id: int) -> Image.Image:
        path = self.coco.loadImgs(_id)[0]["file_name"]
        return Image.open(str(Path(self.root) / path)).convert("RGB")

    def _load_target(self, _id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(_id))

    def _load_images_from_dir(self, index: int) -> Image.Image:
        return Image.open(str(self.paths[index]))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.coco:
            _id = self.ids[index]
            image = self._load_image(_id)
            target = self._load_target(_id)

        else:
            image = self._load_images_from_dir(index)
            target = None

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
