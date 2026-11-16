"""Format MOT: frame,id,x,y,w,h,conf,-1,-1,-1. Wewnętrznie: {frame: ndarray (N, 5) [x1, y1, x2, y2, id]}."""

from __future__ import annotations

from pathlib import Path

import numpy as np

PerFrame = dict[int, np.ndarray]


def to_lines(per_frame: PerFrame) -> list[str]:
    lines = []
    for f in sorted(per_frame):
        for x1, y1, x2, y2, tid in per_frame[f][:, :5]:
            lines.append(f"{f},{int(tid)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,-1,-1,-1")
    return lines


def from_lines(lines: list[str]) -> PerFrame:
    rows: dict[int, list] = {}
    for ln in lines:
        p = ln.strip().split(",")
        if len(p) < 6:
            continue
        f, tid = int(p[0]), int(p[1])
        x, y, w, h = map(float, p[2:6])
        rows.setdefault(f, []).append([x, y, x + w, y + h, tid])
    return {f: np.array(v, dtype=np.float64) for f, v in rows.items()}


def save(per_frame: PerFrame, path: str | Path) -> None:
    Path(path).write_text("\n".join(to_lines(per_frame)) + "\n")


def load(path: str | Path) -> PerFrame:
    return from_lines(Path(path).read_text().splitlines())


def to_dataframe(per_frame: PerFrame):
    """pandas.DataFrame z kolumnami frame, id, x1, y1, x2, y2."""
    import pandas as pd

    rows = [(f, int(r[4]), *r[:4]) for f in sorted(per_frame) for r in per_frame[f]]
    return pd.DataFrame(rows, columns=["frame", "id", "x1", "y1", "x2", "y2"])
