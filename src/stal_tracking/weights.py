"""Jednorazowe pobranie OSNet (Deep-EIoU, sports_model.pth.tar-60) i konwersja do TorchScript
skryptami ARC-SORT z przypiętego commita."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

DEEP_EIOU_FOLDER_ID = "1wItcb0yeGaxOS08_G9yRWBTnpVf0vZ2w"  # oficjalny folder z README Deep-EIoU
ARCSORT_REPO = "https://github.com/Athlitix/arcsort.git"
ARCSORT_COMMIT = "0fe370dbcc0db06e49d4a02757257ad9e0ebfb31"  # z tego commita są też pliki w _vendor/


def _sh(cmd: str, cwd: Path) -> None:
    print("$", cmd)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-2000:])
    r.check_returncode()


def _checkout_arcsort(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    _sh(f"git init -q && git remote add origin {ARCSORT_REPO} && "
        f"git fetch -q --depth 1 origin {ARCSORT_COMMIT} && git checkout -q FETCH_HEAD", cwd=dest)
    return dest


def ensure_reid_weights(weights_dir: str | Path) -> Path:
    weights_dir = Path(weights_dir)
    out = weights_dir / "sports_osnet_traced.pt"
    if out.exists():
        print("Wagi ReID już są:", out)
        return out

    weights_dir.mkdir(parents=True, exist_ok=True)
    raw = weights_dir / "sports_model.pth.tar-60"
    clean = weights_dir / "sports_osnet_clean.pth"

    if not raw.exists():
        import gdown

        files = gdown.download_folder(id=DEEP_EIOU_FOLDER_ID, skip_download=True, quiet=True)
        reid = [f for f in files if f.path.endswith("sports_model.pth.tar-60")]
        if not reid:
            raise RuntimeError(
                "Nie znalazłem sports_model.pth.tar-60. Pobierz ręcznie z "
                f"https://drive.google.com/drive/folders/{DEEP_EIOU_FOLDER_ID} do {weights_dir}"
            )
        gdown.download(id=reid[0].id, output=str(raw), quiet=False)

    with tempfile.TemporaryDirectory() as tmp:
        repo = _checkout_arcsort(Path(tmp) / "arcsort")
        _sh(f'python tools/extract_state_dict.py --input "{raw}" --output "{clean}"', cwd=repo)
        _sh(f'PYTHONPATH=. python tools/convert_osnet_weights.py --input "{clean}" --output "{out}"', cwd=repo)
    return out
