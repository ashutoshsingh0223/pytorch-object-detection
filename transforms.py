from typing import List, Dict, Optional, Tuple

from PIL import Image

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

    left = abs((new_w - w) // 2)
    top = abs((new_h - h) // 2)
    padded_image.paste(image, (left, top))

    return padded_image


class PILToTensor(nn.Module):
    def forward(self, image: 'Image',
                target: Optional[Dict[str, 'Tensor']] = None) -> Tuple['Tensor', Optional[Dict[str, 'Tensor']]]:
        image = F.pil_to_tensor(image)
        return image, target


class ResizeImageAspectRatioPreserve(nn.Module):

    def __init__(self, size: Tuple[int, int], padding_color: Tuple[int, int, int] = (128, 128, 128)):
        super(ResizeImageAspectRatioPreserve, self).__init__()

        self.size = size
        self.padding_color = padding_color

    def forward(self, image: 'Image', target: Optional[Dict[str, 'Tensor']] = None) -> Tuple[
        'Image', Optional[Dict[str, Tensor]]]:
        padded_image = resize(image, self.size, self.padding_color)
        return padded_image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: 'Image', target: Optional[Dict[str, 'Tensor']]) -> Tuple[
        'Tensor', Optional[Dict[str, 'Tensor']]]:
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
