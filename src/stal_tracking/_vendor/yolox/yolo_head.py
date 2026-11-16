# Derived from https://github.com/hsiangwei0903/Deep-EIoU/blob/2fc7d270eec8290fda075b976b5e06683c788e1e/Deep-EIoU/yolox/models/yolo_head.py
# (YOLOX, Megvii, Apache-2.0). Modified: inference-only (training/loss code removed).
# Module/attribute names are unchanged so the original state_dict loads 1:1.

import torch
import torch.nn as nn

from .network_blocks import BaseConv, DWConv


class YOLOXHead(nn.Module):
    def __init__(self, num_classes, width=1.0, strides=(8, 16, 32),
                 in_channels=(256, 512, 1024), act="silu", depthwise=False):
        super().__init__()
        self.n_anchors = 1
        self.num_classes = num_classes
        self.strides = list(strides)
        Conv = DWConv if depthwise else BaseConv
        c = int(256 * width)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        for ch in in_channels:
            self.stems.append(BaseConv(int(ch * width), c, ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(Conv(c, c, 3, 1, act=act), Conv(c, c, 3, 1, act=act)))
            self.reg_convs.append(nn.Sequential(Conv(c, c, 3, 1, act=act), Conv(c, c, 3, 1, act=act)))
            self.cls_preds.append(nn.Conv2d(c, self.n_anchors * num_classes, 1, 1, 0))
            self.reg_preds.append(nn.Conv2d(c, 4, 1, 1, 0))
            self.obj_preds.append(nn.Conv2d(c, self.n_anchors, 1, 1, 0))

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            x = self.stems[k](x)
            cls_out = self.cls_preds[k](self.cls_convs[k](x))
            reg_feat = self.reg_convs[k](x)
            reg_out = self.reg_preds[k](reg_feat)
            obj_out = self.obj_preds[k](reg_feat)
            outputs.append(torch.cat([reg_out, obj_out.sigmoid(), cls_out.sigmoid()], 1))

        hw = [o.shape[-2:] for o in outputs]
        out = torch.cat([o.flatten(start_dim=2) for o in outputs], dim=2).permute(0, 2, 1)
        return self._decode(out, hw)

    def _decode(self, outputs, hw):
        grids, strides = [], []
        for (h, w), s in zip(hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            strides.append(torch.full((*grid.shape[:2], 1), s))
        grids = torch.cat(grids, 1).to(outputs)
        strides = torch.cat(strides, 1).to(outputs)
        outputs = outputs.clone()
        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs  # [B, anchors, 4 (cxcywh) + 1 obj + num_classes]


class YOLOX(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        return self.head(self.backbone(x))
