"""stal-track run CONFIG [--set klucz=wartość ...] [--force detect|track] [--no-render]
stal-track diagnose CONFIG
stal-track weights CONFIG"""

from __future__ import annotations

import argparse

import yaml

from .config import Config


def _overrides(pairs: list[str]) -> dict:
    out = {}
    for p in pairs:
        k, _, v = p.partition("=")
        out[k] = yaml.safe_load(v)  # "0.5" -> float, "true" -> bool, "null" -> None
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="stal-track")
    ap.add_argument("command", choices=["run", "diagnose", "weights"])
    ap.add_argument("config")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL")
    ap.add_argument("--force", nargs="*", default=[], choices=["detect", "track"])
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)

    cfg = Config.from_yaml(a.config, **_overrides(a.set))

    if a.command == "weights":
        from .weights import ensure_reid_weights
        ensure_reid_weights(cfg.weights_dir)
    elif a.command == "diagnose":
        from .detectors import make_detector
        from .tracking import diagnose
        diagnose(cfg, make_detector(cfg.detector, cfg.det_conf))
    else:
        from .pipeline import run
        run(cfg, force=set(a.force), render_video=not a.no_render)


if __name__ == "__main__":
    main()
