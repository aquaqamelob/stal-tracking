"""Detektor YOLOX-X z Deep-EIoU (best_ckpt.pth.tar, trenowany na SportsMOT, 1 klasa: zawodnik).

Bez repo Deep-EIoU, bez Pythona 3.7, bez kompilacji: architektura jest w _vendor/yolox,
preprocessing i postprocessing odtworzone 1:1 z tools/demo.py Deep-EIoU.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Z yolox_x_ch_sportsmot.py (exp Deep-EIoU)
DEPTH, WIDTH, NUM_CLASSES = 1.33, 1.25, 1
TEST_SIZE = (800, 1440)  # (h, w)
NMS_THRESH = 0.7
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_model():
    import torch.nn as nn

    from ._vendor.yolox import YOLOX, YOLOPAFPN, YOLOXHead

    in_channels = [256, 512, 1024]
    model = YOLOX(
        YOLOPAFPN(DEPTH, WIDTH, in_channels=in_channels),
        YOLOXHead(NUM_CLASSES, WIDTH, in_channels=in_channels),
    )
    for m in model.modules():  # jak init_yolo() w exp YOLOX
        if isinstance(m, nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    return model


def load_model(weights: str | Path, device: str, fp16: bool = True, fuse: bool = True):
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    from ._vendor.yolox.network_blocks import BaseConv

    model = build_model()
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)  # strict: błąd = złe wagi
    model.eval()
    if fuse:  # conv+bn -> conv (to samo co --fuse w demo Deep-EIoU)
        for m in model.modules():
            if type(m) is BaseConv and hasattr(m, "bn"):
                m.conv = fuse_conv_bn_eval(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.fuseforward
    model = model.to(device)
    return model.half() if fp16 and device.startswith("cuda") else model


def preproc(img_bgr: np.ndarray, size: tuple[int, int] = TEST_SIZE) -> tuple[np.ndarray, float]:
    """Letterbox (pad 114) -> BGR->RGB -> /255 -> ImageNet mean/std -> CHW."""
    h, w = img_bgr.shape[:2]
    r = min(size[0] / h, size[1] / w)
    resized = cv2.resize(img_bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size[0], size[1], 3), 114.0, dtype=np.float32)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    padded = padded[:, :, ::-1] / 255.0
    padded = (padded - RGB_MEAN) / RGB_STD
    return np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32), r


def postprocess(pred, conf: float, nms: float = NMS_THRESH) -> np.ndarray:
    """pred: tensor [anchors, 5 + C] (cxcywh, obj, cls...) -> ndarray (N, 5) [x1, y1, x2, y2, score]."""
    import torchvision

    boxes = pred[:, :4].clone()
    boxes[:, 0] = pred[:, 0] - pred[:, 2] / 2
    boxes[:, 1] = pred[:, 1] - pred[:, 3] / 2
    boxes[:, 2] = pred[:, 0] + pred[:, 2] / 2
    boxes[:, 3] = pred[:, 1] + pred[:, 3] / 2
    cls_conf, cls_id = pred[:, 5:].max(1)
    score = pred[:, 4] * cls_conf
    keep = score >= conf
    boxes, score, cls_id = boxes[keep].float(), score[keep].float(), cls_id[keep]
    if boxes.shape[0] == 0:
        return np.empty((0, 5))
    idx = torchvision.ops.batched_nms(boxes, score, cls_id, nms)
    return np.column_stack([boxes[idx].cpu().numpy(), score[idx].cpu().numpy()])


class YOLOXDetector:
    """frame_bgr -> (N, 6) [x1, y1, x2, y2, score, 0] w pikselach oryginalnej klatki."""

    def __init__(self, weights: str | Path, conf: float = 0.05, size=TEST_SIZE,
                 device: str | None = None, fp16: bool = True):
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(weights, self.device, fp16=fp16)
        self.dtype = next(self.model.parameters()).dtype
        self.conf, self.size = conf, tuple(size)

    def __call__(self, frame_bgr: np.ndarray) -> np.ndarray:
        import torch

        h, w = frame_bgr.shape[:2]
        img, r = preproc(frame_bgr, self.size)
        x = torch.from_numpy(img).unsqueeze(0).to(self.device, self.dtype)
        with torch.inference_mode():
            pred = self.model(x)[0]
        det = postprocess(pred, self.conf)
        if len(det) == 0:
            return np.empty((0, 6))
        det[:, :4] /= r
        det[:, [0, 2]] = det[:, [0, 2]].clip(0, w)
        det[:, [1, 3]] = det[:, [1, 3]].clip(0, h)
        det = det[(det[:, 2] > det[:, 0] + 1) & (det[:, 3] > det[:, 1] + 1)]  # puste boxy psuły crop ReID
        return np.column_stack([det, np.zeros(len(det))])
