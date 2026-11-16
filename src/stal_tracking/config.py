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

    # Detekcja – niski próg, żeby drugi etap BYTE widział słabe detekcje.
    detector: str = "rfdetr-medium"  # rfdetr-medium | rfdetr-large | yolo
    det_conf: float = 0.05

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

    trail_length_s: float = 1.2

    # ---- wczytywanie ----
    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "Config":
        data = yaml.safe_load(Path(path).read_text()) or {}
        data.update(overrides)
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Nieznane klucze w configu: {sorted(unknown)}")
        return cls(**data)

    # ---- pomocnicze ----
    @staticmethod
    def frames(seconds: float, fps: float, minimum: int = 1) -> int:
        return max(minimum, round(seconds * fps))

    @property
    def stem(self) -> str:
        return Path(self.source).stem

    def out(self, suffix: str) -> Path:
        p = Path(self.output_dir) / f"{self.stem}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reid_weights(self) -> Path:
        return Path(self.weights_dir) / "sports_osnet_traced.pt"

    def key(self, *names: str) -> str:
        """Krótki hash wybranych pól – klucz cache, żeby zmiana parametru nie czytała starych wyników."""
        d = asdict(self)
        payload = json.dumps({n: d[n] for n in names}, sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()[:8]

    # Pola, od których zależy dany etap (cache unieważnia się automatycznie)
    DETECT_FIELDS = ("source", "detector", "det_conf")
    TRACK_FIELDS = DETECT_FIELDS + (
        "preset", "preset_fps", "track_high", "new_track", "track_buffer_s", "use_reid",
    )
