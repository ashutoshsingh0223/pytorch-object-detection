from typing import List, Dict, Any

import torch
import torchvision


def calculate_iou_loss(
    target_boxes: torch.Tensor,
    predicted_boxes: torch.Tensor,
    device="cpu",
    ciou: bool = False,
    giou: bool = False,
):
    if ciou:
        loss = torchvision.ops.ciou_loss(predicted_boxes, target_boxes)
    elif giou:
        loss = torchvision.ops.giou_loss(predicted_boxes, target_boxes)
    else:
        loss = 1 - torchvision.ops.box_iou(predicted_boxes, target_boxes)

    loss = loss.to(device)
    loss.requires_grad = True

    return loss


def build_targets(
    predictions: List[torch.Tensor],
    targets: torch.Tensor,
    model: torch.nn.Module,
    hyperparams: Dict[str, Any],
):
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
    # Example: [[img_id, class, x, y, w, h]], x, y, w, h all are ratios in terms of image size here

    anchor_ids = (
        torch.arange(na, device=targets.device, dtype=targets.dtype)
        .view(na, 1)
        .repeat(1, num_targets)
    )

    t_indices, t_boxes, t_anchors, t_classes = [], [], [], []

    # shape(targets) -> [nt, 6]
    # shape(targets.repeat(na,1,1)) -> [na, nt, 6]
    # shape(anchor_ids) -> [na, nt], shape(anchor_ids[:, :, None]) -> [na, nt, 1]
    # shape(targets) -> [na, nt, 7]
    targets = torch.cat((targets.repeat(na, 1, 1), anchor_ids[:, :, None]), dim=2)
    scale = torch.ones(7, dtype=targets.dtype, device=targets.device)

    # shape(predictions) -> [batch, #anchors, grid_y, grid_x, #classes + 5]

    # Prepare targets for each yolo layer
    for layer_idx, yolo_layer in enumerate(model.yolo_layers):
        # Get anchors and get scale them down to feature map space
        # shape(anchors) -> [na, 2]
        anchors = yolo_layer.anchors / yolo_layer.stride
        layer_predictions = predictions[layer_idx]

        # a tensor =with value[batch, #anchors, grid_y, grid_x, #classes + 5] -> repeat rows for any many targets, take all rows and 3 and 2nd column i.e grid_x, grid_y
        scale[2:6] = torch.tensor(layer_predictions.shape)[[3, 2, 3, 2]]

        # scale targets from ratios(yolo target format) to actual values in feature space
        t = targets * scale

        # Now we need to find target to anchor association. Each target can be associated to one/multiple/no anchors
        # Get rid of anchor-target pairs where the divergence in their width or height is as big as `target_anchor_ratio`.
        # And lose the anchor dimention - the anchors are identified by the anchor-id appended to targets.
        # if num_targets:
        #     r = t[:, :, 4:6] / anchors[:, None, :]
        #     j = (
        #         torch.max(r, 1 / r).max(2)[0] < hyperparams["target_anchor_ratio"]
        #     )  # compare
        #     t = t[j]
        # else:
        #     t = targets[0]

        t = t.reshape(-1, 7)

        images_ids, classes = t[:, 0:2].long().T

        # Actual coordinates and wh values in feature map space
        xy = t[:, 2:4]
        wh = t[:, 4:6]

        # Cell associations to a grid in feature map scape x=1.2, y=3.4 => i=1, j=3 for association with grid cell (1,3)
        ij = xy.long()
        i, j = xy.T

        # anchor ids
        a_ids = t[:, 6].long()

        # clamp grid cell indices to avoid overflow from feature map scale
        t_indices.append(
            (
                images_ids,
                a_ids,
                j.clamp_(0, scale[3].long() - 1).long(),
                i.clamp_(0, scale[2].long() - 1).long(),
            )
        )

        # Subject cell associations again to get proper regression targets, we already have grid associations in t_indices
        t_boxes.append(torch.cat((xy - ij, wh), 1))

        t_anchors.append(anchors[a_ids])
        t_classes.append(classes)

    return t_indices, t_boxes, t_anchors, t_classes


def calculate_loss(
    predictions: List[torch.Tensor],
    targets: torch.Tensor,
    model: torch.nn.Module,
    hyperparams: Dict[str, Any],
):
    # shape(predictions) -> [batch, #anchors, grid_y, grid_x, #5+num_classes]
    # shape(targets) -> [batch, 6], ([img_id, class, x, y, w, h])
    cls_loss = torch.nn.CrossEntropyLoss()
    obj_loss = torch.nn.BCEWithLogitsLoss()

    t_indices, t_boxes, t_anchors, t_classes = build_targets(
        predictions, targets, model, hyperparams
    )

    for layer_idx, layer_predictions in enumerate(predictions):
        layer_t_indices, layer_t_boxes = t_indices[layer_idx], t_boxes[layer_idx]
        layer_t_anchors, layer_t_classes = t_anchors[layer_idx], t_boxes[layer_idx]

        image_ids, anch_ids, j, i = layer_t_indices
        predicted_box = layer_predictions[image_ids, anch_ids, j, i]

        # Sigmoid to predicted x, y offsets.
        predicted_xy = predicted_box[:, 0:2].sigmoid()
        predcited_wh = torch.exp(predicted_box[:, 2:4]) * layer_t_anchors

        predicted_box = torch.cat((predicted_xy, predcited_wh), 1)
