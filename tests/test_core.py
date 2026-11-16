"""Testy części bez GPU/modeli: format MOT, przeliczanie czasu, interpolacja i renumeracja ID."""

import numpy as np
import pytest

from stal_tracking import mot
from stal_tracking.config import Config
from stal_tracking.postprocess import postprocess


def _cfg(**kw):
    return Config(source="x.mp4", output_dir="/tmp/out", **kw)


def test_mot_roundtrip():
    pf = {1: np.array([[10, 20, 50, 120, 7]], float), 3: np.array([[12, 22, 52, 122, 7]], float)}
    back = mot.from_lines(mot.to_lines(pf))
    assert back.keys() == pf.keys()
    np.testing.assert_allclose(back[3], pf[3])


@pytest.mark.parametrize("fps,expected", [(25, 10), (50, 20), (30, 12)])
def test_seconds_to_frames(fps, expected):
    assert Config.frames(0.4, fps) == expected


def test_interpolation_fills_short_gap_and_renumbers():
    # track 42 widoczny w klatkach 1 i 4 (dziura 2) -> interpolacja dopisuje 2 i 3, ID -> 1
    pf = {1: np.array([[0, 0, 10, 10, 42]], float), 4: np.array([[30, 0, 40, 10, 42]], float)}
    out = postprocess(_cfg(), pf, embeddings={}, fps=25)
    assert sorted(out) == [1, 2, 3, 4]
    assert {int(r[4]) for v in out.values() for r in v} == {1}
    np.testing.assert_allclose(out[2][0, :4], [10, 0, 20, 10])


def test_long_gap_not_interpolated():
    pf = {1: np.array([[0, 0, 10, 10, 1]], float), 30: np.array([[0, 0, 10, 10, 1]], float)}
    out = postprocess(_cfg(), pf, embeddings={}, fps=25)
    assert sorted(out) == [1, 30]


def test_unknown_config_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("source: a.mp4\noutput_dir: o\nmerge_trhesh: 0.9\n")
    with pytest.raises(ValueError, match="merge_trhesh"):
        Config.from_yaml(p)


def test_cache_key_changes_with_params():
    a, b = _cfg(), _cfg(track_high=0.5)
    assert a.key(*Config.DETECT_FIELDS) == b.key(*Config.DETECT_FIELDS)
    assert a.key(*Config.TRACK_FIELDS) != b.key(*Config.TRACK_FIELDS)
