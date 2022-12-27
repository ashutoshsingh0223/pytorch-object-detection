from typing import List, Dict, Optional, Tuple

from PIL import Image
import cv2

import numpy as np

import torch
from torch import nn, Tensor

from torchvision.transforms import functional as F, InterpolationMode, transforms as T


def resize(image: 'Image', size: Tuple[int, int], padding_color: Tuple[int, int, int] = (128, 128, 128)):
    im_width, im_height = image.size
    w, h = size

    scale_factor = min(w / im_width, h / im_height)
    new_w = int(im_width * scale_factor)
    new_h = int(im_height * scale_factor)

    image = image.resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
    padded_image = Image.new(image.mode, (w, h), padding_color)

    # This abs is not required you could just write ((w - new_w) / 2)
    left = abs((new_w - w) // 2)
    top = abs((new_h - h) // 2)
    padded_image.paste(image, (left, top))

    return padded_image


class PILToTensor(nn.Module):
    def forward(self, image: 'Image',
                target: Optional[Dict[str, 'Tensor']] = None) -> Tuple['Tensor', Optional[Dict[str, 'Tensor']]]:
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


# Already managed by ConvertImageDtype

# class ScaleImage(nn.Module):
#     def __init__(self, scale: float = 255.0) -> None:
#         super().__init__()
#         self.scale = scale
#
#     def forward(
#         self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
#     ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
#         image = image.div(self.scale)
#         return image, target


class ResizeImageAspectRatioPreserve(nn.Module):

    def __init__(self, size: Tuple[int, int], padding_color: Tuple[int, int, int] = (128, 128, 128)):
        super(ResizeImageAspectRatioPreserve, self).__init__()

        self.size = size
        self.padding_color = padding_color

    def forward(self, image: 'Image', target: Optional[Dict[str, 'Tensor']] = None) -> Tuple[
        'Image', Optional[Dict[str, Tensor]]]:

        # padded_image = resize(image, self.size, self.padding_color)
        def letterbox_image(img, inp_dim):
            '''resize image with unchanged aspect ratio using padding'''
            img_w, img_h = img.shape[1], img.shape[0]
            w, h = inp_dim
            new_w = int(img_w * min(w / img_w, h / img_h))
            new_h = int(img_h * min(w / img_w, h / img_h))
            resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

            canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w,
            :] = resized_image

            return canvas

        def prep_image(img, inp_dim):
            """
            Prepare image for inputting to the neural network.

            Returns a Variable
            """
            img = letterbox_image(img, (inp_dim, inp_dim))
            img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).float().div(255.0)
            return img
        img = prep_image(image, 416)
        return img, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: 'Image', target: Optional[Dict[str, 'Tensor']]
                 ) -> Tuple['Tensor', Optional[Dict[str, 'Tensor']]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
