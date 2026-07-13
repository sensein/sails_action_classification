"""
slow_fast experiments/common/data.py does `from common.labels import ...`
(bare, self-referencing package style). See feature_extraction/conftest.py
for why an autouse fixture (not just a one-time collection-time fix) is
needed to keep sys.modules["common"] pointed at the right package
throughout this suite.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SLOWFAST_EXPERIMENTS_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sailsprep" / "action_model_testing" / "slow_fast" / "experiments"
)


def _fix_common_shadow() -> None:
    for _name in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_name]
    if str(SLOWFAST_EXPERIMENTS_ROOT) in sys.path:
        sys.path.remove(str(SLOWFAST_EXPERIMENTS_ROOT))
    sys.path.insert(0, str(SLOWFAST_EXPERIMENTS_ROOT))


_fix_common_shadow()


@pytest.fixture(autouse=True)
def _fix_common_shadow_per_test():
    _fix_common_shadow()
    yield
