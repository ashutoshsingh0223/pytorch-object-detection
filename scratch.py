from PIL import Image
import torch
from YoloV3 import YoloV3
from presets import YOLOTrain
import numpy as np
from YoloV3 import loss

model = YoloV3.YoloV3("YoloV3/cfg/yolov3.cfg")

i = Image.open("./test_images/dog-cycle-car.png")
d = YOLOTrain(size=(416, 416))
img, _ = d(i)
img = img.unsqueeze(0)
results = model(img)

targets = np.loadtxt("test_images/test.txt")
boxes = torch.zeros((targets.shape[0], targets.shape[1] + 1))
boxes[:, 1:] = torch.from_numpy(targets)
