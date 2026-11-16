"""Jedna konfiguracja całego pipeline'u. Czasy w sekundach, przeliczane na klatki wg FPS wideo."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    source: str
    output_dir: str
    weights_dir: str = "weights"

    # Fragment nagrania: od start_s przez duration_s sekund (None = do końca)
    start_s: float = 0.0
    duration_s: float | None = None

    # Detekcja – niski próg, żeby drugi etap BYTE widział słabe detekcje.
    detector: str = "yolox-sportsmot"  # yolox-sportsmot | rfdetr-medium | rfdetr-large | yolo
    det_conf: float = 0.05
    yolox_size: tuple[int, int] = (800, 1440)  # (h, w) jak w Deep-EIoU; mniejsze = szybciej, gorzej
    fp16: bool = True

    # Tracker – domyślne z presetu; autorzy: obniżanie pogarsza HOTA.
    preset: str = "sportsmot"
    preset_fps: float = 25.0
    track_high: float = 0.6
    new_track: float = 0.7
    track_buffer_s: float = 1.2  # 30 klatek @ 25 FPS
    use_reid: bool = True

    # Post-processing (wartości z artykułu ARC-SORT)
    interp_max_gap_s: float = 0.4  # 10 klatek @ 25 FPS
    split_thresh: float = 0.45
    split_persist: int = 5
    merge_thresh: float = 0.85
    merge_max_gap_s: float = 12.0  # 300 klatek @ 25 FPS
    merge_motion_tol: float | None = None  # None = bez bramki ruchu

    # ---- wczytywanie ----
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update(overrides)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Nieznane klucze w configu: {sorted(unknown)}")
        if "yolox_size" in data:
            data["yolox_size"] = tuple(data["yolox_size"])
        return cls(**data)

    # ---- pomocnicze ----
    @staticmethod
    def frames(seconds: float, fps: float, minimum: int = 1) -> int:
        return max(minimum, round(seconds * fps))

    @property
    def stem(self) -> str:
        stem = Path(self.source).stem
        if self.start_s or self.duration_s is not None:
            end = "end" if self.duration_s is None else f"{self.start_s + self.duration_s:g}"
            stem += f"_{self.start_s:g}-{end}s"
        return stem

    def frame_range(self, fps: float, n_frames: int) -> range:
        """Numery klatek (od 1, jak w MOT) objęte fragmentem start_s..start_s+duration_s."""
        first = round(self.start_s * fps) + 1
        last = n_frames if self.duration_s is None else min(n_frames, first - 1 + round(self.duration_s * fps))
        if first > last:
            raise ValueError(f"Pusty fragment: start_s={self.start_s}, duration_s={self.duration_s}, "
                             f"wideo ma {n_frames / fps:.1f} s")
        return range(first, last + 1)

    def out(self, suffix: str) -> Path:
        p = Path(self.output_dir) / f"{self.stem}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reid_weights(self) -> Path:
        return Path(self.weights_dir) / "sports_osnet_traced.pt"

    @property
    def yolox_weights(self) -> Path:
        return Path(self.weights_dir) / "best_ckpt.pth.tar"

    def key(self, *names: str) -> str:
        """Krótki hash wybranych pól – klucz cache, żeby zmiana parametru nie czytała starych wyników."""
        d = asdict(self)
        payload = json.dumps({n: d[n] for n in names}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    # Pola, od których zależy dany etap (cache unieważnia się automatycznie)
    DETECT_FIELDS = ("source", "start_s", "duration_s", "detector", "det_conf", "yolox_size", "fp16")
    TRACK_FIELDS = DETECT_FIELDS + (
        "preset", "preset_fps", "track_high", "new_track", "track_buffer_s", "use_reid",
    )
