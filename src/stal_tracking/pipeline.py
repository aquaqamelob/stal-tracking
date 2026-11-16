"""detect -> track -> postprocess -> ID. Drogie etapy mają cache kluczowany parametrami."""

from __future__ import annotations

from . import mot
from .config import Config
from .detectors import detect_video, make_detector
from .postprocess import postprocess
from .tracking import track_video


def track(cfg: Config, force: set[str] = frozenset()) -> mot.PerFrame:
    """Zwraca {klatka: ndarray (N, 5) [x1, y1, x2, y2, id]} i zapisuje plik MOT obok."""
    det_cache = cfg.out(f".dets.{cfg.key(*Config.DETECT_FIELDS)}.npz")
    trk_cache = cfg.out(f".online.{cfg.key(*Config.TRACK_FIELDS)}.pkl")
    if "detect" in force:
        det_cache.unlink(missing_ok=True)
    if force & {"detect", "track"}:
        trk_cache.unlink(missing_ok=True)

    # Detektor ładujemy tylko, gdy faktycznie trzeba liczyć detekcje
    detector = None if det_cache.exists() else make_detector(cfg)
    dets = detect_video(detector, cfg, cache=det_cache)

    online = track_video(cfg, dets, cache=trk_cache)
    final = postprocess(cfg, online.tracks, online.embeddings, online.fps)

    mot_path = cfg.out("_tracks.txt")
    mot.save(final, mot_path)
    n_ids = len({int(t) for v in final.values() for t in v[:, 4]})
    print(f"Unikalnych ID: {n_ids} | MOT: {mot_path}")
    return final


run = track  # zgodność wstecz
