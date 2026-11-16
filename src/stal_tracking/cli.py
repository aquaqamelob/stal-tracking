"""stal-track run CONFIG [--set klucz=wartość ...] [--force detect|track]
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
    a = ap.parse_args(argv)

    cfg = Config.from_yaml(a.config, **_overrides(a.set))

    if a.command == "weights":
        from .weights import ensure_weights
        ensure_weights(cfg)
    elif a.command == "diagnose":
        from .detectors import make_detector
        from .tracking import diagnose
        diagnose(cfg, make_detector(cfg))
    else:
        from .pipeline import track
        track(cfg, force=set(a.force))


if __name__ == "__main__":
    main()
