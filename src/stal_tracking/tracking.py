"""Etap 2: ARC-SORT online na (zcache'owanych) detekcjach + zbieranie embeddingów do offline merge."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .config import Config
from .detectors import Detector
from .video import iter_frames, probe


@dataclass
class OnlineResult:
    fps: float
    n_frames: int
    tracks: dict[int, np.ndarray] = field(default_factory=dict)          # {f: (N,5+) x1y1x2y2,id}
    embeddings: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    camera_motion: dict[int, np.ndarray | None] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "OnlineResult":
        return pickle.loads(path.read_bytes())  # własny cache, nie dane z zewnątrz


def make_tracker(cfg: Config, fps: float, reid: bool):
    from arcsort import ARCTracker

    buf = Config.frames(cfg.track_buffer_s, fps)
    return ARCTracker.from_preset(
        cfg.preset,
        reid_weights=str(cfg.reid_weights) if reid else None,
        track_buffer=buf,
        arbitration={"window": buf},
        track_high_thresh=cfg.track_high,
        new_track_thresh=cfg.new_track,
    )


def reid_available(cfg: Config) -> bool:
    ok = cfg.use_reid and cfg.reid_weights.exists()
    if cfg.use_reid and not ok:
        print("UWAGA: brak wag ReID – tryb motion-only, bez offline merge.")
    return ok


def track_video(cfg: Config, dets: dict[int, np.ndarray], cache: Path | None = None) -> OnlineResult:
    if cache is not None and cache.exists():
        print("Tracki online z cache:", cache)
        return OnlineResult.load(cache)

    info = probe(cfg.source)
    frames = cfg.frame_range(info.fps, info.n_frames)
    reid = reid_available(cfg)
    tracker = make_tracker(cfg, info.fps, reid)
    res = OnlineResult(fps=info.fps, n_frames=0)

    for f, frame in tqdm(iter_frames(cfg.source, frames), total=len(frames), desc="Tracking"):
        t = tracker.update(dets.get(f, np.empty((0, 6))), frame=frame)
        res.camera_motion[f] = tracker.camera_motion
        res.n_frames = f
        if len(t):
            res.tracks[f] = t.copy()
            if reid:  # embeddingi boxów tracków do offline split/merge
                boxes = t[:, :4].astype(np.float64)
                embs = tracker.embedding_computer.compute_embeddings(frame, boxes)
                res.embeddings[f] = (boxes, embs.astype(np.float32))

    if cache is not None:
        res.save(cache)
    return res


def diagnose(cfg: Config, detector: Detector, n_frames: int = 60) -> None:
    """Szybki test: czy są detekcje i czy tracker cokolwiek zwraca (bez ReID)."""
    info = probe(cfg.source)
    start = cfg.frame_range(info.fps, info.n_frames).start
    tracker = make_tracker(cfg, info.fps, reid=False)
    n_det, n_trk, confs = [], [], []
    for _, frame in iter_frames(cfg.source, range(start, min(start + n_frames, info.n_frames + 1))):
        d = detector(frame)
        n_det.append(len(d))
        n_trk.append(len(tracker.update(d, frame=frame)))
        confs.extend(d[:, 4].tolist())
    confs = np.array(confs)
    print(f"Klatek: {len(n_det)} | osób/klatkę: śr. {np.mean(n_det):.1f} | "
          f"tracków/klatkę: śr. {np.mean(n_trk):.1f} (ostatnie 10: {n_trk[-10:]})")
    if not len(confs):
        print("BRAK DETEKCJI osób – problem z detektorem/ścieżką wideo, nie z trackerem.")
        return
    print(f"Pewność detekcji: mediana {np.median(confs):.2f} | "
          f">= {cfg.track_high}: {np.mean(confs >= cfg.track_high):.0%} | "
          f">= {cfg.new_track}: {np.mean(confs >= cfg.new_track):.0%}")
    if np.mean(n_trk[-10:]) == 0:
        print(f"Detekcje są, tracków brak: za mało detekcji >= {cfg.new_track}. Obniż track_high/new_track.")
    else:
        print("OK – można puszczać całość.")
