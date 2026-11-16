# stal-tracking

Śledzenie zawodników z nagrań meczów aż do **ID na każdej klatce**:
**YOLOX-X (Deep-EIoU, SportsMOT) → ARC-SORT online z OSNet ReID → interpolacja + split/merge po wyglądzie offline**.

Modele z Deep-EIoU działają tu na zwykłym Pythonie 3.10+/aktualnym torch – bez repo Deep-EIoU,
bez Pythona 3.7, bez `cython_bbox` i bez patchowania `demo.py`.

## Użycie w swoim notebooku (Colab)

```python
!pip install -q "stal-tracking[yolox] @ git+https://github.com/aquaqamelob/stal-tracking.git"

from google.colab import drive; drive.mount('/content/drive')
from stal_tracking import Config, ensure_weights, make_detector, diagnose, track, render

cfg = Config(
    source="/content/drive/MyDrive/stal/1/stal_5_minut.mp4",
    output_dir="/content/drive/MyDrive/stal/1/out",
    weights_dir="/content/drive/MyDrive/stal/weights",
    duration_s=60,                     # opcjonalnie: tylko fragment (start_s=... przesuwa początek)
)
ensure_weights(cfg)                    # raz: YOLOX + ReID na Drive
diagnose(cfg, make_detector(cfg))      # opcjonalnie: 60 klatek, sanity check
ids = track(cfg)                       # {klatka: ndarray (N,5) [x1,y1,x2,y2,id]} + plik MOT
render(cfg, ids)                       # opcjonalnie: wideo z ID -> output_dir/<nazwa>_tracks.mp4
```

Fragment ma własne pliki wyjściowe i cache (np. `mecz_0-60s_tracks.txt`), numery klatek są liczone od początku całego nagrania.

Do pandas: `from stal_tracking.mot import to_dataframe; df = to_dataframe(ids)`.
Aktualizacja kodu: ta sama linijka `pip install` z `--force-reinstall --no-deps`.

Pojedyncze komponenty też da się brać osobno:

```python
from stal_tracking.yolox import YOLOXDetector          # sam detektor: frame_bgr -> (N,6)
det = YOLOXDetector(cfg.yolox_weights, conf=0.05)
```

## Modele

| plik w `weights_dir` | co to | skąd |
|---|---|---|
| `best_ckpt.pth.tar` | YOLOX-X, 1 klasa (zawodnik), trenowany na SportsMOT, wejście 800×1440 | Deep-EIoU |
| `sports_model.pth.tar-60` → `sports_osnet_traced.pt` | OSNet x1.0 ReID trenowany na sporcie, konwersja do TorchScript | Deep-EIoU + skrypty ARC-SORT |

## Struktura

```
src/stal_tracking/
  config.py       # jedna dataclass z parametrami (sekundy, nie klatki) + YAML
  yolox.py        # detektor YOLOX z Deep-EIoU: preproc/postproc 1:1 z ich demo.py
  detectors.py    # rejestr detektorów (yolox-sportsmot, rfdetr-*, yolo) + detekcja wideo z cache
  tracking.py     # ARC-SORT online + embeddingi do merge; diagnose()
  postprocess.py  # interpolacja, split, merge, renumeracja ID
  pipeline.py     # track(): detect -> track -> postprocess -> ID
  rendering.py    # render(): opcjonalne wideo z ID i śladami
  mot.py          # format MOT, to_dataframe
  weights.py      # pobranie modeli
  cli.py          # stal-track run|diagnose|weights CONFIG
  _vendor/        # ARC-SORT tools (Apache-2.0, commit 0fe370d), YOLOX inference (Apache-2.0)
```

## Cache

`detect` i `track` zapisują wyniki w `output_dir` z hashem parametrów w nazwie.
Zmiana `merge_thresh` → tylko post-processing (sekundy). Zmiana `track_high` → tracking od nowa,
detekcje z cache. Zmiana detektora / `det_conf` / `yolox_size` → wszystko. Wymuszenie: `track(cfg, force={"detect"})`.

## CLI

```bash
pip install -e ".[yolox,dev]"
stal-track weights configs/stal_5_minut.yaml
stal-track run configs/stal_5_minut.yaml --set merge_thresh=0.9
pytest
```

## Licencje

`_vendor/`: [Athlitix/arcsort](https://github.com/Athlitix/arcsort) i YOLOX (Megvii) przez
[Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU) – Apache-2.0. Opcja `yolo` = Ultralytics (AGPL-3.0).
Wagi modeli pochodzą od autorów Deep-EIoU; repo ich nie zawiera.
