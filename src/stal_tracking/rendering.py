"""Opcjonalny render: wideo z boxami, ID i śladami ruchu (kompensacja ruchu kamery z cache trackingu)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from . import mot
from .config import Config
from .video import iter_frames, probe


def render(cfg: Config, ids: mot.PerFrame | None = None, out: str | Path | None = None,
           h264: bool = True) -> Path:
    """ids: wynik track(cfg); None = wczytaj plik MOT z output_dir.
    h264=True: przekodowanie ffmpeg, żeby wideo grało w przeglądarce/Drive (mp4v z OpenCV często nie gra)."""
    from arcsort.visualizer import Visualizer

    if ids is None:
        if not cfg.mot_path.exists():
            raise FileNotFoundError(f"Brak {cfg.mot_path} – najpierw ids = track(cfg)")
        ids = mot.load(cfg.mot_path)

    camera_motion: dict = {}
    if cfg.track_cache.exists():
        from .tracking import OnlineResult
        camera_motion = OnlineResult.load(cfg.track_cache).camera_motion
    else:
        print("Brak cache trackingu – ślady bez kompensacji ruchu kamery.")

    info = probe(cfg.source)
    frames = cfg.frame_range(info.fps, info.n_frames)
    out = Path(out) if out else cfg.video_path
    use_ffmpeg = h264 and shutil.which("ffmpeg") is not None
    raw = out.with_suffix(".mp4v.mp4") if use_ffmpeg else out

    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), info.fps, (info.width, info.height))
    viz = Visualizer(trace_length=Config.frames(cfg.trail_length_s, info.fps), motion_compensated_trails=True)
    try:
        for f, frame in tqdm(iter_frames(cfg.source, frames), total=len(frames), desc="Render"):
            tr = ids.get(f, np.empty((0, 5)))
            writer.write(viz.annotate(frame, tr, camera_motion=camera_motion.get(f)))
    finally:
        writer.release()

    if use_ffmpeg:
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            raw.unlink()
        else:
            print("ffmpeg nie przeszedł, zostaje wersja mp4v:", r.stderr[-500:])
            out = raw
    print("Wideo:", out)
    return out
