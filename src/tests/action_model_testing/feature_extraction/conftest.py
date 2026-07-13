"""
feature_extraction extractors do `from common.bbox import ...` (bare,
sibling-package style). Some of this suite's tests import the extractor
lazily inside a test function rather than at module top-level, so a
one-time collection-time fix isn't enough — other action_model_testing
suites' conftest.py files run during the same collection pass and can
re-poison sys.modules["common"] afterward. An autouse fixture re-applies
the fix before every test in this directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

FEATURE_EXTRACTION_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sailsprep" / "action_model_testing" / "feature_extraction"
)


def _fix_common_shadow() -> None:
    for _name in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_name]
    if str(FEATURE_EXTRACTION_ROOT) in sys.path:
        sys.path.remove(str(FEATURE_EXTRACTION_ROOT))
    sys.path.insert(0, str(FEATURE_EXTRACTION_ROOT))


_fix_common_shadow()


@pytest.fixture(autouse=True)
def _fix_common_shadow_per_test():
    _fix_common_shadow()
    yield
