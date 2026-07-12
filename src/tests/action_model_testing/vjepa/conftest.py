"""
vjepa/clips_fixed_length/vjepa_clip_level_ablation.py and
vjepa/clips_without_coi_crop/{locomotion,rmm}/train_probe_ablation.py do
`from common.probes import ...` / `from common.bbox_utils import ...`
(bare, self-referencing package style, resolved via each script's own
sys.path.insert(0, .../vjepa) at import time). See
action_model_testing/feature_extraction/conftest.py for why an autouse
fixture (not just a one-time collection-time purge) is needed.
"""
from __future__ import annotations

import sys

import pytest


def _fix_common_shadow() -> None:
    for _name in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_name]


_fix_common_shadow()


@pytest.fixture(autouse=True)
def _fix_common_shadow_per_test():
    _fix_common_shadow()
    yield
