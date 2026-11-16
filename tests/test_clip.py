"""Wycinanie fragmentu: zakres klatek, seek w wideo, nazwy plików wyjściowych."""

import cv2
import numpy as np
import pytest

from stal_tracking.config import Config
from stal_tracking.video import iter_frames


def _cfg(**kw):
    return Config(source="/x/mecz.mp4", output_dir="/tmp/out", **kw)


def test_frame_range_minute_at_25fps():
    r = _cfg(duration_s=60).frame_range(25, 15000)
    assert (r.start, r[-1], len(r)) == (1, 1500, 1500)


def test_frame_range_offset_and_clamp():
    r = _cfg(start_s=590, duration_s=60).frame_range(25, 15000)
    assert (r.start, r[-1]) == (14751, 15000)


def test_frame_range_empty_raises():
    with pytest.raises(ValueError):
        _cfg(start_s=700).frame_range(25, 15000)


def test_full_video_by_default():
    assert len(_cfg().frame_range(25, 15000)) == 15000
    assert _cfg().stem == "mecz"


def test_clip_gets_own_output_names_and_cache_keys():
    full, clip = _cfg(), _cfg(duration_s=60)
    assert clip.stem == "mecz_0-60s"
    assert _cfg(start_s=120, duration_s=60).stem == "mecz_120-180s"
    assert full.key(*Config.DETECT_FIELDS) != clip.key(*Config.DETECT_FIELDS)


@pytest.fixture
def numbered_video(tmp_path):
    """Wideo, w którym jasność klatki f (od 1) = 2*f – da się sprawdzić, którą klatkę dostaliśmy."""
    path = str(tmp_path / "v.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
    for f in range(1, 101):
        w.write(np.full((48, 64, 3), 2 * f, np.uint8))
    w.release()
    return path


def test_iter_frames_range_returns_right_frames(numbered_video):
    got = list(iter_frames(numbered_video, range(41, 51)))
    assert [f for f, _ in got] == list(range(41, 51))
    for f, img in got:
        assert abs(img.mean() - 2 * f) < 3


def test_iter_frames_all(numbered_video):
    assert [f for f, _ in iter_frames(numbered_video)] == list(range(1, 101))
