from __future__ import division

from typing import List
import time
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import cv2
import pandas as pd

from util import *

import os
import os.path as osp

import pickle as pkl
import random

from YoloV3.YoloV3 import YoloV3
from YoloV3.utils import transform_detections


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
    parse.add_argument("--objects", help='File path for objects names', required=False)
    parser.add_argument("--batch-size", help="Batch size", default=1, type=int)
    parser.add_argument("--confidence", help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--nms-threshold", help="NMS threshold", default=0.4, type=float)
    parser.add_argument("--cfg", help="Config file", default="YoloV3/cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", help= "weights file for yolo", default="yolov3.weights", type=str)
    parser.add_argument("--resolution",
                        help="Input resolution of the network."
                             "Increase to increase accuracy. Decrease to increase speed", default=416, type=int)

    return parser.parse_args()


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

#If there's a GPU available, put the model on GPU
if CUDA:
    model.cuda()
#Set the model in evaluation mode
model.eval()