from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _open(path: str) -> cv2.VideoCapture:
    p = Path(path)
    if not p.exists():
        parent = next((d for d in [p.parent, *p.parents] if d.exists()), None)
        videos = sorted(x.name for x in parent.iterdir() if x.suffix.lower() in VIDEO_EXT) if parent else []
        hint = f" Wideo w {parent}: {videos}" if videos else (f" Najbliższy istniejący folder: {parent}" if parent else "")
        raise FileNotFoundError(f"Plik nie istnieje: {path}.{hint}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Plik istnieje, ale OpenCV nie umie go otworzyć: {path} "
                           f"(kodek? sprawdź: !ffprobe -hide_banner \"{path}\")")
    return cap


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    n_frames: int


def probe(path: str) -> VideoInfo:
    cap = _open(path)
    info = VideoInfo(
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps=cap.get(cv2.CAP_PROP_FPS) or 25.0,
        n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )
    cap.release()
    return info


def iter_frames(path: str, frames: range | None = None) -> Iterator[tuple[int, np.ndarray]]:
    """Klatki BGR numerowane od 1 (konwencja MOT), opcjonalnie tylko z zakresu `frames`."""
    cap = _open(path)
    try:
        f = 0
        if frames is not None and frames.start > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frames.start - 1)
            f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if f != frames.start - 1:  # kontener nie seekuje dokładnie -> czytamy po kolei
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                f = 0
        stop = None if frames is None else frames.stop - 1
        while stop is None or f < stop:
            ok, frame = cap.read()
            if not ok:
                break
            f += 1
            if frames is None or f >= frames.start:
                yield f, frame
    finally:
        cap.release()
