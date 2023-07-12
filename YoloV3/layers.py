from typing import Union, Tuple
import torch.nn as nn
import torch


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors, classes):
        super(DetectionLayer, self).__init__()

        self.anchors = torch.FloatTensor(anchors)
        self.num_anchors = len(self.anchors)
        self.classes = classes
        
        self.num_attrs = self.classes + 5

        self.image_size = None

        self.stride = None

    @staticmethod
    def make_grid(grid_size, num_anchors, dtype, device):
        grid = torch.arange(grid_size)
        x, y = torch.meshgrid(grid, grid)

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        # Concat to form (x,y) offsets and repeat each offset as many times as num_anchors, add one more dimention to
        # handle directly adding to prediction tensor(for each pred in entire batch)
        grid_cell_offsets = torch.cat((x, y), 1).repeat(1, num_anchors).reshape(-1, 2).unsqueeze(0).to(dtype=dtype, device=device)
        return grid_cell_offsets

    def forward(self, x, image_size):
        bs, ch, fm_h, fm_w = x.shape

        self.image_size = image_size
        self.stride = self.image_size // fm_w
        # x will have shape - [batch_size, (#classes + #bbox_attrs) * #anchors, fm_h, fm_w]
        # change the shape to [batch_size, #anchors, fm_h, fm_w, #classes + #bbox_attrs] 
        x = x.view(bs, self.num_anchors, self.num_attrs, fm_h, fm_w)
        x = x.permute(0, 1, 3, 4, 2).view(bs, self.num_anchors * fm_h * fm_w, self.num_attrs).contiguous()

        if not self.training:
            cell_offsets = DetectionLayer.make_grid(grid_size=fm_h, num_anchors=self.num_anchors, dtype=x.dtype, device=x.device)

            # Bring anchor to feature map coordinates. Repeat tensor of 3 anchors for every pixel in grid
            anchors = self.anchors / self.stride
            anchor_grid = anchors.repeat((fm_h * fm_w, 1)).unsqueeze(0).to(x.device)
    
            x[..., 0:2] = x[..., 0:2].sigmoid() + cell_offsets  # xy
            x[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_grid  # wh
            x[..., 4:] = x[..., 4:].sigmoid() # conf, cls

            x[..., :4] *= self.stride

        x = x.view(bs, self.num_anchors, fm_h, fm_w, self.num_attrs)
        return x
