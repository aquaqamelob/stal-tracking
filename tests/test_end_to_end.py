"""track() + render() na syntetycznym wideo z podstawionym detektorem (bez GPU i wag)."""

import cv2
import numpy as np
import pytest

pytest.importorskip("arcsort")


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "mecz.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)  # tekstura, żeby CMC miało punkty
    for f in range(1, 76):
        w.write(bg)
    w.release()
    return path


def fake_detector(frame):
    fake_detector.f += 1
    x = 20 + 2 * fake_detector.f  # dwóch "zawodników" idących w prawo
    return np.array([[x, 50, x + 30, 130, 0.9, 0], [x, 140, x + 30, 220, 0.85, 0]], dtype=float)


def test_track_then_render_clip(video, tmp_path, monkeypatch):
    import stal_tracking.pipeline as pipeline
    from stal_tracking import Config, render

    fake_detector.f = 0
    monkeypatch.setattr(pipeline, "make_detector", lambda cfg: fake_detector)
    cfg = Config(source=video, output_dir=str(tmp_path / "out"), use_reid=False, start_s=0.4, duration_s=2)

    ids = pipeline.track(cfg)
    assert min(ids) == 11 and max(ids) <= 60          # 0.4 s..2.4 s @ 25 FPS
    assert {int(r[4]) for v in ids.values() for r in v} == {1, 2}
    assert cfg.mot_path.exists() and cfg.det_cache.exists() and cfg.track_cache.exists()

    out = render(cfg, ids)
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 50
    cap.release()

    # drugi przebieg: detekcje i tracking z cache, detektor nie jest wołany
    monkeypatch.setattr(pipeline, "make_detector", lambda cfg: pytest.fail("detektor nie powinien ruszyć"))
    assert pipeline.track(cfg).keys() == ids.keys()
