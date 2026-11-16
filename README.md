# stal-tracking

Śledzenie zawodników z nagrań meczów: **RF-DETR → ARC-SORT (online) → interpolacja + split/merge po wyglądzie (offline) → render**.

## Struktura

```
src/stal_tracking/
  config.py       # jedna dataclass z parametrami (sekundy, nie klatki) + wczytywanie YAML
  detectors.py    # RF-DETR / YOLO za jednym kontraktem: frame -> (N,6); detekcja całego wideo z cache
  tracking.py     # ARC-SORT online + embeddingi do merge; diagnose()
  postprocess.py  # interpolacja, split, merge, renumeracja ID
  render.py       # wideo z trajektoriami
  pipeline.py     # detect -> track -> postprocess -> render
  mot.py          # zapis/odczyt formatu MOT
  weights.py      # jednorazowe pobranie i konwersja OSNet
  cli.py          # stal-track run|diagnose|weights CONFIG
  _vendor/        # 2 skrypty z ARC-SORT (Apache-2.0), przypięte do commita 0fe370d
configs/          # jeden YAML na nagranie
notebooks/colab.ipynb
tests/
```

## Cache

Etapy `detect` i `track` zapisują wyniki obok wyjścia, z hashem parametrów w nazwie pliku.
Zmiana `merge_thresh` → liczy się tylko post-processing (sekundy).
Zmiana `track_high` → tracking od nowa, detekcje z cache.
Zmiana detektora / `det_conf` → wszystko od nowa.
Wymuszenie: `--force detect` / `--force track`.

## Uruchomienie

Colab: `notebooks/colab.ipynb` (podmień login w komórce z `REPO`).

Lokalnie / na serwerze z GPU:

```bash
pip install -e ".[rfdetr,weights,dev]"
stal-track weights configs/stal_5_minut.yaml
stal-track diagnose configs/stal_5_minut.yaml
stal-track run configs/stal_5_minut.yaml --set merge_thresh=0.9 --no-render
pytest
```

## Licencje

Kod z `_vendor/` pochodzi z [Athlitix/arcsort](https://github.com/Athlitix/arcsort) (Apache-2.0).
Opcja `yolo` ciągnie Ultralytics (AGPL-3.0).
