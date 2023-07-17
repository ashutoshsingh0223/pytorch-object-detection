from typing import List, Dict, Optional, Tuple, Any, Callable

from PIL import Image
import cv2

import numpy as np

import torch
from torch import nn, Tensor

from torchvision.transforms import functional as F, transforms as T


def resize(
    image: Image.Image,
    size: Tuple[int, int],
    padding_color: Tuple[int, int, int] = (128, 128, 128),
):
    im_width, im_height = image.size
    w, h = size

    scale_factor = min(w / im_width, h / im_height)
    new_w = int(im_width * scale_factor)
    new_h = int(im_height * scale_factor)

    image = image.resize((new_w, new_h), resample=Image.Resampling.CUBIC)
    padded_image = Image.new(image.mode, (w, h), padding_color)

    # This abs is not required you could just write ((w - new_w) / 2)
    left = abs((new_w - w) // 2)
    top = abs((new_h - h) // 2)
    padded_image.paste(image, (left, top))

    return padded_image


class AbsoluteBoxes(nn.Module):
    def __init_(self):
        pass

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None):
        if target is not None:
            relative_boxes = target["boxes"]
            scale = torch.tensor(image.shape)[[2, 1, 2, 1]]
            target["boxes"] = relative_boxes * scale
        return image, target


class RelativeBoxes(nn.Module):
    def __init_(self):
        pass

    def forward(self, image: Tensor, target: Optional[Dict[str, Tensor]] = None):
        if target is not None:
            absolute_boxes = target["boxes"]
            scale = torch.tensor(image.shape)[[2, 1, 2, 1]]
            target["boxes"] = absolute_boxes / scale
        return image, target


class PILToTensor(nn.Module):
    def forward(
        self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class ResizeImageAspectRatioPreserve(nn.Module):
    def __init__(
        self,
        size: int,
        padding_color: Tuple[int, int, int] = (128, 128, 128),
    ):
        super(ResizeImageAspectRatioPreserve, self).__init__()

        self.size = size
        self.padding_color = padding_color

    def forward(
        self, image: Image.Image, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Image.Image, Optional[Dict[str, Tensor]]]:
        # padded_image = resize(image, self.size, self.padding_color)
        def letterbox_image(img, inp_dim):
            """resize image with unchanged aspect ratio using padding"""
            img_w, img_h = img.shape[1], img.shape[0]
            w, h = inp_dim
            new_w = int(img_w * min(w / img_w, h / img_h))
            new_h = int(img_h * min(w / img_w, h / img_h))
            resized_image = cv2.resize(
                img, (new_w, new_h), interpolation=cv2.INTER_CUBIC
            )

            canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

            canvas[
                (h - new_h) // 2 : (h - new_h) // 2 + new_h,
                (w - new_w) // 2 : (w - new_w) // 2 + new_w,
                :,
            ] = resized_image

            return canvas

        def prep_image(img, inp_dim):
            """
            Prepare image for inputting to the neural network.

            Returns a Variable
            """
            img = np.array(img)
            img = letterbox_image(img, (inp_dim, inp_dim))
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).float().div(255.0)
            return img

        img = prep_image(image, self.size)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(
        self, image: Image.Image, target: Optional[Dict[str, Tensor]]
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ConvertCocoPolysToMask:
    def __call__(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[Image.Image, Dict[str, Tensor]]:
        """
        Convert coco like annotations to targets as Dict[str, Tensor]. This handles tranform of bboxes, segmentations, keypoints(shape).

        This is used as a dataset transform in `coco_utils.get_coco` method

        Args:
            image (Image.Image): Segmentations from dataset
            target (Dict[str, Any]): height of mask

        Returns:
            Tuple[Image.Image, Dict[str, Tensor]]: Tuple of image and transformed targets.
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno: List[Dict[str, Any]] = target["annotations"]
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        target = {}

        classes = torch.tensor([obj["category_id"] for obj in anno], dtype=torch.int64)

        # TODO: If keypoints and category id are given for keypoint detection task, we can draw a approximate bounding box.
        boxes = torch.as_tensor(
            [obj["bbox"] for obj in anno], dtype=torch.float32
        ).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # TODO: If segmentation is absent but bbox is present and keypoint detection is to be done only for one 1 category
        # then segmentation can be set equal to bbox
        masks = None
        if anno and "segmentation" in anno[0]:
            masks = convert_coco_poly_to_mask(
                [obj["segmentation"] for obj in anno], h, w
            )

        # TODO: Handle both the above mentioned TODOs in dataset generation and finalization step.

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = torch.as_tensor(
                [obj["keypoints"] for obj in anno], dtype=torch.float32
            )
            num_detections = keypoints.shape[0]
            if num_detections:
                keypoints = keypoints.view(num_detections, -1, 3)

        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        area = area[keep]
        classes = classes[keep]
        iscrowd = iscrowd[keep]

        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target["boxes"] = boxes

        target["labels"] = classes
        target["image_id"] = image_id

        if keypoints is not None:
            target["keypoints"] = keypoints
        if masks is not None:
            target["masks"] = masks

        # for conversion to coco api

        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def check_transforms(transforms: Callable):
    if hasattr(transforms, "transforms"):
        t = transforms.transforms
        if isinstance(t, Compose):
            return True
        else:
            return False
    else:
        return False
