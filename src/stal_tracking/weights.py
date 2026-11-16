"""Jednorazowe pobranie modeli z Deep-EIoU (Google Drive autorów) do cfg.weights_dir:

- best_ckpt.pth.tar        – YOLOX-X trenowany na SportsMOT (detektor zawodników), używany bez konwersji
- sports_model.pth.tar-60  – OSNet x1.0 ReID trenowany na sporcie -> TorchScript dla ARC-SORT
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from .config import Config

# ID plików z oficjalnego folderu Deep-EIoU (1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w)
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w"
YOLOX_FILE_ID = "1834kh10-X0Tu743fgmN7jXPVDKgq4ZqR"
REID_FILE_ID = "14zzlm1nI9Ws_Il9RYNChwPC7Fsul7xwl"

ARCSORT_REPO = "https://github.com/Athlitix/arcsort.git"
ARCSORT_COMMIT = "0fe370dbcc0db06e49d4a02757257ad9e0ebfb31"  # z tego commita są też pliki w _vendor/


def _sh(cmd: str, cwd: Path) -> None:
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    r.check_returncode()


def _download(file_id: str, out: Path) -> Path:
    if out.exists():
        return out
    import gdown

    out.parent.mkdir(parents=True, exist_ok=True)
    got = gdown.download(id=file_id, output=str(out), quiet=False)
    if not got or not out.exists():
        raise RuntimeError(f"Nie udało się pobrać {out.name} (limit Google Drive?). "
                           f"Pobierz ręcznie z {DRIVE_FOLDER} do {out.parent}")
    return out


def ensure_yolox_weights(cfg: Config) -> Path:
    return _download(YOLOX_FILE_ID, cfg.yolox_weights)


def ensure_reid_weights(cfg: Config) -> Path:
    out = cfg.reid_weights
    if out.exists():
        return out
    raw = _download(REID_FILE_ID, out.parent / "sports_model.pth.tar-60")
    clean = out.parent / "sports_osnet_clean.pth"
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "arcsort"
        repo.mkdir()
        _sh(f"git init -q && git remote add origin {ARCSORT_REPO} && "
            f"git fetch -q --depth 1 origin {ARCSORT_COMMIT} && git checkout -q FETCH_HEAD", cwd=repo)
        _sh(f'python tools/extract_state_dict.py --input "{raw}" --output "{clean}"', cwd=repo)
        _sh(f'PYTHONPATH=. python tools/convert_osnet_weights.py --input "{clean}" --output "{out}"', cwd=repo)
    return out


def ensure_weights(cfg: Config) -> None:
    """Pobiera tylko to, czego wymaga config (YOLOX gdy detector=yolox-sportsmot, ReID gdy use_reid)."""
    if cfg.detector == "yolox-sportsmot":
        print("YOLOX:", ensure_yolox_weights(cfg))
    if cfg.use_reid:
        print("ReID:", ensure_reid_weights(cfg))
