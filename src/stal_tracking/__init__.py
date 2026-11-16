"""Publiczne API:

    from stal_tracking import Config, ensure_weights, make_detector, diagnose, track
"""

from .config import Config

__all__ = ["Config", "diagnose", "ensure_weights", "make_detector", "track"]
__version__ = "0.2.0"


def __getattr__(name):  # leniwe importy: `import stal_tracking` nie ciągnie torch/cv2
    if name == "track":
        from .pipeline import track
        return track
    if name == "ensure_weights":
        from .weights import ensure_weights
        return ensure_weights
    if name == "make_detector":
        from .detectors import make_detector
        return make_detector
    if name == "diagnose":
        from .tracking import diagnose
        return diagnose
    raise AttributeError(name)
