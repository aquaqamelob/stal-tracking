"""Detektory osób. Kontrakt: Detector(frame_bgr) -> ndarray (N, 6) [x1, y1, x2, y2, score, cls=0]."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from .config import Config
from .video import iter_frames, probe

Detector = Callable[[np.ndarray], np.ndarray]
EMPTY = np.empty((0, 6))


def person_mask(dets) -> np.ndarray:
    """Pretrenowany RF-DETR zwraca class_id w numeracji COCO (person = 1), a class_names
    liczy od 0 – więc filtrujemy po nazwie z predict(), awaryjnie po class_id == 1."""
    names = dets.data.get("class_name") if hasattr(dets, "data") else None
    if names is not None and len(names) == len(dets):
        return np.array([str(n).lower() == "person" for n in names], dtype=bool)
    return np.asarray(dets.class_id) == 1


def _rfdetr(size: str, conf: float) -> Detector:
    from arcsort.interop import from_supervision
    from rfdetr import RFDETRLarge, RFDETRMedium

    model = RFDETRLarge() if size == "large" else RFDETRMedium()
    try:
        model.optimize_for_inference()
    except Exception as e:  # noqa: BLE001 – opcjonalna optymalizacja
        print("optimize_for_inference pominięte:", e)

    def detect(frame_bgr: np.ndarray) -> np.ndarray:
        dets = model.predict(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), threshold=conf)
        if len(dets) == 0:
            return EMPTY
        out = from_supervision(dets[person_mask(dets)])
        out[:, 5] = 0  # ARC-SORT jest jednoklasowy
        return out

    return detect


def _yolo(conf: float) -> Detector:
    # UWAGA: Ultralytics to AGPL-3.0 – istotne, jeśli repo/aplikacja jest publiczna.
    from arcsort.detection import create_detector

    return create_detector("yolo11x.pt", imgsz=1280, conf_thresh=conf, classes=[0]).detect


def _yolox_sportsmot(cfg: Config) -> Detector:
    from .yolox import YOLOXDetector

    if not cfg.yolox_weights.exists():
        raise FileNotFoundError(f"Brak {cfg.yolox_weights} – uruchom najpierw ensure_weights(cfg).")
    return YOLOXDetector(cfg.yolox_weights, conf=cfg.det_conf, size=cfg.yolox_size, fp16=cfg.fp16)


DETECTORS: dict[str, Callable[[Config], Detector]] = {
    "yolox-sportsmot": _yolox_sportsmot,
    "rfdetr-medium": lambda c: _rfdetr("medium", c.det_conf),
    "rfdetr-large": lambda c: _rfdetr("large", c.det_conf),
    "yolo": lambda c: _yolo(c.det_conf),
}


def make_detector(cfg: Config) -> Detector:
    try:
        factory = DETECTORS[cfg.detector.lower()]
    except KeyError:
        raise ValueError(f"Nieznany detektor: {cfg.detector}. Dostępne: {list(DETECTORS)}") from None
    return factory(cfg)


# ---------- etap 1: detekcja całego wideo z cache na dysku ----------

def save_detections(dets: dict[int, np.ndarray], path: Path, n_frames: int) -> None:
    rows = [np.column_stack([np.full(len(d), f), d]) for f, d in dets.items() if len(d)]
    arr = np.vstack(rows) if rows else np.empty((0, 7))
    np.savez_compressed(path, dets=arr, n_frames=n_frames)


def load_detections(path: Path) -> tuple[dict[int, np.ndarray], int]:
    z = np.load(path)
    arr, n = z["dets"], int(z["n_frames"])
    out = {f: EMPTY for f in range(1, n + 1)}
    for f in np.unique(arr[:, 0]).astype(int):
        out[f] = arr[arr[:, 0] == f, 1:]
    return out, n


def detect_video(detector: Detector, source: str, cache: Path | None = None) -> dict[int, np.ndarray]:
    """Najdroższy etap. Z cache: strojenie trackera/merge'a nie odpala detektora ponownie."""
    if cache is not None and cache.exists():
        print("Detekcje z cache:", cache)
        return load_detections(cache)[0]

    info = probe(source)
    dets: dict[int, np.ndarray] = {}
    for f, frame in tqdm(iter_frames(source), total=info.n_frames, desc="Detekcja"):
        dets[f] = detector(frame)
        if f == 100 and not any(len(d) for d in dets.values()):
            raise RuntimeError("0 detekcji osób w 100 klatkach – uruchom diagnose() i sprawdź detektor.")
    if cache is not None:
        save_detections(dets, cache, n_frames=len(dets))
    return dets
