from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .config import Config
from .mot import PerFrame
from .video import iter_frames, probe


def render(
    source: str,
    out_video: Path,
    tracks: PerFrame,
    camera_motion: dict[int, np.ndarray | None],
    trail_length_s: float,
) -> None:
    from arcsort.visualizer import Visualizer

    info = probe(source)
    writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    viz = Visualizer(trace_length=Config.frames(trail_length_s, info.fps), motion_compensated_trails=True)
    try:
        for f, frame in tqdm(iter_frames(source), total=info.n_frames, desc="Render"):
            tr = tracks.get(f, np.empty((0, 5)))
            writer.write(viz.annotate(frame, tr, camera_motion=camera_motion.get(f)))
    finally:
        writer.release()
