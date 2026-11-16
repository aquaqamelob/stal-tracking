from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int


def probe(path: str) -> VideoInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć {path}")
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str, limit: int | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Klatki BGR numerowane od 1 (konwencja MOT)."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Nie mogę otworzyć {path}")
    try:
        f = 0
        while limit is None or f < limit:
            ok, frame = cap.read()
            if not ok:
                break
            f += 1
            yield f, frame
    finally:
        cap.release()
