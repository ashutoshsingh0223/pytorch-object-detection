from typing import List, Any, Optional, Union, Tuple
from pathlib import Path
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision.transforms import functional as F, ToPILImage
from torchvision.utils import draw_bounding_boxes

from PIL import Image

import numpy as np

import pickle as pkl
import random

from YoloV3.YoloV3 import YoloV3
from YoloV3.utils import transform_detections
from datasets.coco import get_coco
import presets
from base_constants import TRAIN, VALIDATION, TEST


def load_classes(namesfile: str) -> List[str]:
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def arg_parse():
    """
    Parse arguements to the detect module
    """

    parser = argparse.ArgumentParser(description="YOLO v3 Detection Module")

    parser.add_argument(
        "--images",
        help="Image / Directory containing images to perform detection upon",
        type=str,
    )
    parser.add_argument(
        "--det", help="Image / Directory to store detections to", type=str
    )
    parser.add_argument(
        "--annotations-file",
        help="Path to annotations file",
        type=str,
        default=None,
        required=False,
    )
    parser.add_argument(
        "--data-augmentation",
        help="Data Augmentation to use",
        default="default",
        type=str,
    )
    parser.add_argument("--classes", help="File path for class names", required=False)
    parser.add_argument("--batch-size", help="Batch size", default=1, type=int)
    parser.add_argument(
        "--confidence",
        help="Object Confidence to filter predictions",
        default=0.5,
        type=float,
    )
    parser.add_argument(
        "--nms-threshold", help="NMS threshold", default=0.4, type=float
    )
    parser.add_argument(
        "--cfg", help="Config file", default="./YoloV3/cfg/yolov3.cfg", type=str
    )
    parser.add_argument("--workers", help="Number of workers", default=2, type=int)
    parser.add_argument(
        "--weights",
        help="weights file for yolo",
        default="./weights/yolov3_coco_80_weights/yolov3.weights",
        type=str,
    )
    parser.add_argument(
        "--resolution",
        help="Input resolution of the network."
        "Increase to increase accuracy. Decrease to increase speed",
        default=416,
        type=int,
    )

    return parser.parse_args()


def get_dataset(
    name: str,
    datapath: Union["Path", str],
    mode: str = TRAIN,
    transforms: Optional[Any] = None,
    annotations_file: Optional[str] = None,
):
    index = {"coco": get_coco}

    data_fn = index[name]
    dataset = data_fn(
        image_dir=datapath,
        mode=mode,
        annotations_file=annotations_file,
        transforms=transforms,
    )
    return dataset


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrainResize(
            data_augmentation=args.data_augmentation,
            size=(args.resolution, args.resolution),
        )
    # elif args.weights and args.test_only:
    #     weights = torchvision.models.get_weight(args.weights)
    #     trans = weights.transforms()
    #     return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEvalResize(
            size=(args.resolution, args.resolution)
        )


def draw_bbox(
    bboxes: torch.Tensor,
    feature_map_res: Tuple[int, int],
    raw_image_path: Union[str, "Path"],
    classes: List[str],
):
    raw_image = Image.open(raw_image_path)
    im_width, im_height = raw_image.size
    w, h = feature_map_res
    scale_factor = min(w / im_width, h / im_height)

    padding_w = (w - (im_width * scale_factor)) / 2
    padding_h = (h - (im_height * scale_factor)) / 2

    # Subtract padding from bounding box coordinates
    # x co-ordinates of lower left and upper right corner
    bboxes[:, [1, 3]] -= padding_w
    # y co-ordinates -- similar
    bboxes[:, [2, 4]] -= padding_h

    # Now scale the boxes to bring co-ordinates on the same level as raw image
    bboxes[:, [1, 3]] /= scale_factor
    bboxes[:, [2, 4]] /= scale_factor

    # clip any bounding boxes that have any corner outside the image
    bboxes[:, [1, 3]] = torch.clip(bboxes[:, [1, 3]], min=0.0, max=float(im_width))
    bboxes[:, [2, 4]] = torch.clip(bboxes[:, [2, 4]], min=0.0, max=float(im_height))

    raw_image = F.pil_to_tensor(raw_image)

    labels = [classes[int(x[-1])] for x in bboxes]
    raw_image = draw_bounding_boxes(raw_image, bboxes[:, 1:5], labels=labels)

    raw_image = ToPILImage(raw_image)


if __name__ == "__main__":
    args = arg_parse()
    images = args.images
    batch_size = args.batch_size
    confidence = args.confidence
    nms_threshold = args.nms_threshold
    start = 0
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    num_classes = 80  # For COCO
    classes = load_classes(args.classes)

    print("Starting weights load")
    model = YoloV3(args.cfg)
    model.load_weights(args.weights)
    print("Finishing weights load")

    model.net_info["height"] = args.resolution
    input_dim = int(model.net_info["height"])
    assert input_dim % 32 == 0
    assert input_dim > 32

    # If there's a GPU available, put the model on GPU
    model.to(device)
    # Set the model in evaluation mode
    model.eval()

    print("Starting data load")

    train_dataset = get_dataset(
        "coco", datapath=args.images, mode=TRAIN, transforms=get_transform(True, args)
    )
    val_dataset = get_dataset(
        "coco",
        datapath=args.images,
        mode=VALIDATION,
        transforms=get_transform(False, args),
    )
    test_dataset = get_dataset(
        "coco", datapath=args.images, mode=TEST, transforms=get_transform(False, args)
    )

    train_loader = DataLoader(
        dataset=train_dataset, num_workers=args.workers, shuffle=True
    )
    val_loader = DataLoader(
        dataset=val_dataset, num_workers=args.workers, shuffle=False
    )
    test_loader = DataLoader(
        dataset=test_dataset, num_workers=args.workers, shuffle=False
    )
    print("Finish data load")

    write = False
    for batch_idx, (batch, target) in enumerate(val_loader):
        batch = batch.to(device)

        with torch.no_grad():
            predictions = model(batch, target, device=device)

        predictions = transform_detections(
            predictions, confidence, num_classes, nms_threshold
        )

        images_idx_start = batch_idx * batch_size
        image_idx_end = images_idx_start + batch_size - 1

        if predictions is None:
            continue

        else:
            # predictions has image idx(within batch) as the first value for every row(bounding box attrs)
            # Convert this within batch index to index according to dataloader
            predictions[:, 0] += batch_idx * batch_size
            if not write:
                output = predictions
                write = True
            output = torch.cat((output, predictions), 0)

        # Iterate over batch and get all predictions for an image in that batch

        for image_idx in range(
            images_idx_start, min(image_idx_end + 1, len(val_loader.dataset))
        ):
            # Get detections(predictions) for this image_idx
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == image_idx]
            print(objs)
