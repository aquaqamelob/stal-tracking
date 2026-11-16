"""Orkiestracja: detect -> track -> postprocess -> render. Każdy drogi etap ma cache kluczowany parametrami."""

from __future__ import annotations

from . import mot
from .config import Config
from .detectors import detect_video, make_detector
from .postprocess import postprocess
from .render import render
from .tracking import track_video


def run(cfg: Config, force: set[str] = frozenset(), render_video: bool = True) -> mot.PerFrame:
    det_cache = cfg.out(f".dets.{cfg.key(*Config.DETECT_FIELDS)}.npz")
    trk_cache = cfg.out(f".online.{cfg.key(*Config.TRACK_FIELDS)}.pkl")
    for stage, path in (("detect", det_cache), ("track", trk_cache)):
        if stage in force or ("detect" in force and stage == "track"):
            path.unlink(missing_ok=True)

    # Detektor ładujemy tylko, gdy faktycznie trzeba liczyć detekcje
    detector = None if det_cache.exists() else make_detector(cfg.detector, cfg.det_conf)
    dets = detect_video(detector, cfg.source, cache=det_cache)

    online = track_video(cfg, dets, cache=trk_cache)
    final = postprocess(cfg, online.tracks, online.embeddings, online.fps)

    mot_path = cfg.out("_tracks.txt")
    mot.save(final, mot_path)
    n_ids = len({int(t) for v in final.values() for t in v[:, 4]})
    print(f"Unikalnych ID: {n_ids} | MOT: {mot_path}")

    if render_video:
        video_path = cfg.out("_arcsort.mp4")
        render(cfg.source, video_path, final, online.camera_motion, cfg.trail_length_s)
        print("Wideo:", video_path)
    return final
