from typing import List, Any, Optional, Union
from pathlib import Path
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import cv2
import pandas as pd

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

    parser = argparse.ArgumentParser(description='YOLO v3 Detection Module')

    parser.add_argument("--images", help="Image / Directory containing images to perform detection upon", type=str)
    parser.add_argument("--det", help="Image / Directory to store detections to", type=str)
    parser.add_argument("--annotations-file", help="Path to annotations file", type=str)
    parser.add_argument("--data-augmentation", help='Data Augmentation to use', default='default', type=str)
    parser.add_argument("--classes", help='File path for class names', required=False)
    parser.add_argument("--batch-size", help="Batch size", default=1, type=int)
    parser.add_argument("--confidence", help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--nms-threshold", help="NMS threshold", default=0.4, type=float)
    parser.add_argument("--cfg", help="Config file", default="YoloV3/cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", help= "weights file for yolo", default="yolov3.weights", type=str)
    parser.add_argument("--resolution",
                        help="Input resolution of the network."
                             "Increase to increase accuracy. Decrease to increase speed", default=416, type=int)

    return parser.parse_args()


def get_dataset(name: str, datapath: Union['Path', str], mode: str = TRAIN, transforms: Optional[List[Any]] = None,
                annotations_file: Optional[str] = None):
    index = {'coco': get_coco}

    data_fn = index[name]
    dataset = data_fn(image_dir=datapath, mode=mode, annotations_file=annotations_file, transforms=transforms)
    return dataset


def get_transform(train, args):
    if train:
        return presets.DetectionPresetTrainResize(data_augmentation=args.data_augmentation,
                                                  size=(args.resolution, args.resolution))
    # elif args.weights and args.test_only:
    #     weights = torchvision.models.get_weight(args.weights)
    #     trans = weights.transforms()
    #     return lambda img, target: (trans(img), target)
    else:
        return presets.DetectionPresetEvalResize(size=(args.resolution, args.resolution))


args = arg_parse()
images = args.images
batch_size = args.batch_size
confidence = args.confidence
nms_threshold = args.nms_threshold
start = 0
CUDA = torch.cuda.is_available()

num_classes = 80    #For COCO
classes = load_classes(args.objects)

model = YoloV3(args.cfg)
model.load_weights(args.weightsfile)

model.net_info["height"] = args.resolution
input_dim = int(model.net_info["height"])
assert input_dim % 32 == 0
assert input_dim > 32

# If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()
# Set the model in evaluation mode
model.eval()

train_dataset = get_dataset('coco', datapath=args.images, mode=TRAIN, transforms=get_transform(True, args))
val_dataset = get_dataset('coco', datapath=args.images, mode=VALIDATION, transforms=get_transform(False, args))
test_dataset = get_dataset('coco', datapath=args.images, mode=TEST, transforms=get_transform(False, args))
