"""Etap 3 (offline): interpolacja krótkich dziur -> split po wyglądzie -> merge po wyglądzie -> nowe ID 1..K."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import mot
from ._vendor.interpolate_tracks import interpolate_sequence
from ._vendor.merge_tracklets import Tracklet, attach_embeddings, merge_tracklets, split_tracklets
from .config import Config


def to_tracklets(per_frame: mot.PerFrame) -> dict[int, Tracklet]:
    tracks: dict[int, Tracklet] = {}
    for f in sorted(per_frame):
        for x1, y1, x2, y2, tid in per_frame[f][:, :5]:
            t = tracks.setdefault(int(tid), Tracklet(int(tid)))
            t.frames.append(f)
            t.boxes.append(np.array([x1, y1, x2, y2]))
    return tracks


def from_tracklets(tracks: dict[int, Tracklet], mapping: dict[int, int]) -> mot.PerFrame:
    renumber = {r: i + 1 for i, r in enumerate(sorted(set(mapping.values())))}
    out = defaultdict(list)
    for tid, t in tracks.items():
        new_id = renumber[mapping[tid]]
        for f, b in zip(t.frames, t.boxes):
            out[f].append([*b, new_id])
    return {f: np.array(v, dtype=np.float64) for f, v in out.items()}


def postprocess(
    cfg: Config,
    online: mot.PerFrame,
    embeddings: dict[int, tuple[np.ndarray, np.ndarray]],
    fps: float,
) -> mot.PerFrame:
    lines = interpolate_sequence(mot.to_lines(online), max_gap=Config.frames(cfg.interp_max_gap_s, fps, minimum=2))
    tracks = to_tracklets(mot.from_lines(lines))
    n_online = len(tracks)

    if not embeddings:
        print("Offline merge pominięty (brak ReID).")
        return from_tracklets(tracks, {t: t for t in tracks})

    tracks = split_tracklets(tracks, embeddings, split_thresh=cfg.split_thresh, persist=cfg.split_persist)
    n_split = len(tracks)
    attach_embeddings(tracks, embeddings)
    mapping = merge_tracklets(
        tracks,
        merge_thresh=cfg.merge_thresh,
        max_gap=Config.frames(cfg.merge_max_gap_s, fps),
        motion_tol=float("inf") if cfg.merge_motion_tol is None else cfg.merge_motion_tol,
    )
    print(f"Offline: {n_online} tracków online -> {n_split} po splicie -> {len(set(mapping.values()))} po scaleniu")
    return from_tracklets(tracks, mapping)
