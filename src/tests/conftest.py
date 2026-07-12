from __future__ import annotations

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch
import fastapi.routing as _fr
import fastapi.utils as _fu

# Ensure src/ is on the path so `sailsprep` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# NOTE: several action_model_testing scripts do a bare `from common.xxx
# import ...` (sibling-package style), each expecting ITS OWN local
# `common/` dir to be resolved. Since Python caches "common" in sys.modules
# under one shared name, adding all of their parent dirs to sys.path here
# would make whichever one is imported first "win" for the rest of the
# session and break every other one. Each conflicting suite instead gets
# its own nested conftest.py (e.g. action_model_testing/Video_Swin/conftest.py)
# that fixes up sys.path and purges the stale "common" cache right before
# its own tests are collected.

# Patch StaticFiles before annotation.py is imported
patch("starlette.staticfiles.StaticFiles.__init__", return_value=None).start()



_real = _fu.create_model_field

def _safe(*args: object, **kwargs: object) -> object:
    try:
        return _real(*args, **kwargs)
    except Exception:
        return MagicMock()

_fr.create_model_field = _safe  # type: ignore[assignment]
_fu.create_model_field = _safe  # type: ignore[assignment]

SOURCE_ROOT = Path(__file__).resolve().parent.parent / "sailsprep"


sys.path.insert(0, str(SOURCE_ROOT / "id_tracking_model" / "target_id" / "child_id"))