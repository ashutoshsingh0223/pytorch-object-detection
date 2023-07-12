from typing import List, Dict

import torch
import torchvision


def calculate_iou_loss(target_boxes: torch.Tensor, predicted_boxes: torch.Tensor, device='cpu', ciou: bool = False, giou: bool = False):
    if ciou:
        loss = torchvision.ops.ciou_loss(predicted_boxes, target_boxes)
    elif giou:
        loss = torchvision.ops.giou_loss(predicted_boxes, target_boxes)
    else:
        loss = 1 - torchvision.ops.box_iou(predicted_boxes, target_boxes)

    loss = loss.to(device)
    loss.requires_grad = True

    return loss


def build_targets(predictions: List[torch.Tensor], targets: torch.Tensor, model: torch.nn.Module):
    """
    1. for each yolo detection layer

    Args:
        predictions (List[torch.Tensor]): _description_
        targets (torch.Tensor): _description_
        model (torch.nn.Module): _description_
    """

    num_targets = targets.shape[0]
    na = model.yolo_layers[0].num_anchors

    # shape of targets - [batch, 6]
    # Example: [[img_id, class, x, y, w, h]], x, y, w, h all are scaled by image size here
    
    anchor_ids = torch.arange(na, device=targets.device, dtype=targets.dtype).view(na, 1).repeat(1, num_targets)

    # shape(targets) -> [nt, 6] 
    # shape(targets.repeat(na,1,1)) -> [na, nt, 6]
    # shape(anchor_ids) -> [na, nt], shape(anchor_ids[:, :, None]) -> [na, nt, 1]
    # shape(targets) -> [na, nt, 7]
    targets = torch.cat((targets.repeat(na, 1, 1), anchor_ids[:, :, None]), dim=2)

    for layer_idx, yolo_layer in enumerate(model.yolo_layers):
        # Get anchors and get scale them down to feature map space
        anchors = yolo_layer.anchors / yolo_layer.stride

        layer_predictions = predictions[layer_idx]
        

        

def calculate_loss(predictions: List[torch.Tensor], targets: torch.Tensor, model: torch.nn.Module):
    cls_loss = torch.nn.CrossEntropyLoss()
    obj_loss = torch.nn.BCEWithLogitsLoss()

    for layer_idx, layer_predictions in enumerate(predictions):
        pass
