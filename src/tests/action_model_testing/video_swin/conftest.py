"""
Video Swin clip_based/sliding_window scripts do `from common.utils import ...`
(bare, sibling-package style). See feature_extraction/conftest.py for why an
autouse fixture (not just a one-time collection-time fix) is needed to keep
sys.modules["common"] pointed at the right package throughout this suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

VIDEO_SWIN_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sailsprep" / "action_model_testing" / "video_swin"
)


def _fix_common_shadow() -> None:
    for _name in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_name]
    if str(VIDEO_SWIN_ROOT) in sys.path:
        sys.path.remove(str(VIDEO_SWIN_ROOT))
    sys.path.insert(0, str(VIDEO_SWIN_ROOT))


_fix_common_shadow()


@pytest.fixture(autouse=True)
def _fix_common_shadow_per_test():
    _fix_common_shadow()
    yield
