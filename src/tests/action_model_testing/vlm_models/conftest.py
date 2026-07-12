"""
vlm_models scripts do `from clips.xxx import ...`, `from common.xxx import ...`
and `from window_classification.xxx import ...` (bare, sibling-package style).

Several unrelated action_model_testing suites also have their own local
`common/` package. Python caches the first-imported one under
sys.modules["common"], and since all of these suites' conftest.py files run
during the same pytest collection pass (in directory-traversal order), a
one-time fix at collection time isn't enough for tests that import lazily
inside a test function — a later suite's conftest can re-poison the cache
before this suite's tests execute. An autouse fixture re-applies the fix
right before every test here too.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

VLM_ROOT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "sailsprep" / "action_model_testing" / "vlm_models"
)


def _fix_common_shadow() -> None:
    for _name in [k for k in sys.modules if k == "common" or k.startswith("common.")]:
        del sys.modules[_name]
    if str(VLM_ROOT) in sys.path:
        sys.path.remove(str(VLM_ROOT))
    sys.path.insert(0, str(VLM_ROOT))


_fix_common_shadow()


@pytest.fixture(autouse=True)
def _fix_common_shadow_per_test():
    _fix_common_shadow()
    yield
