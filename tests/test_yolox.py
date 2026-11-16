"""YOLOX bez prawdziwych wag: architektura, zapis/odczyt checkpointu w formacie Deep-EIoU, kontrakt detektora."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from stal_tracking import yolox  # noqa: E402


def test_forward_shape():
    m = yolox.build_model().eval()
    with torch.inference_mode():
        out = m(torch.zeros(1, 3, 64, 128))
    # anchory: (64/8*128/8) + (64/16*128/16) + (64/32*128/32) = 128 + 32 + 8
    assert out.shape == (1, 168, 4 + 1 + yolox.NUM_CLASSES)


def test_bn_eps_matches_yolox_exp():
    m = yolox.build_model()
    bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
    assert bns and all(b.eps == 1e-3 for b in bns)


def test_detector_contract_with_fake_checkpoint(tmp_path):
    ckpt = tmp_path / "best_ckpt.pth.tar"
    torch.save({"model": yolox.build_model().state_dict(), "start_epoch": 40}, ckpt)
    det = yolox.YOLOXDetector(ckpt, conf=0.0, size=(64, 128), device="cpu", fp16=False)
    frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
    out = det(frame)
    assert out.ndim == 2 and out.shape[1] == 6
    if len(out):
        assert (out[:, [0, 2]] <= 320).all() and (out[:, [1, 3]] <= 180).all()
        assert (out[:, 5] == 0).all()


def test_fused_equals_unfused(tmp_path):
    ckpt = tmp_path / "w.pth"
    m = yolox.build_model().eval()
    for p in m.modules():  # niezerowe statystyki BN, żeby fuzja była coś warta sprawdzenia
        if isinstance(p, torch.nn.BatchNorm2d):
            p.running_mean.uniform_(-0.1, 0.1)
            p.running_var.uniform_(0.5, 1.5)
    torch.save({"model": m.state_dict()}, ckpt)
    a = yolox.load_model(ckpt, "cpu", fp16=False, fuse=False)
    b = yolox.load_model(ckpt, "cpu", fp16=False, fuse=True)
    x = torch.randn(1, 3, 64, 128)
    with torch.inference_mode():
        torch.testing.assert_close(a(x), b(x), rtol=1e-3, atol=1e-3)
