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

# Allow `from clips.xxx import ...` etc.
VLM_ROOT = Path(__file__).parent.parent / "sailsprep" / "action_model_testing" / "vlm_models"
sys.path.insert(0, str(VLM_ROOT))

# window_classifier_ovis/qwen do `from shared_utils import ...` (bare, same-dir style)
sys.path.insert(0, str(VLM_ROOT / "window_classification"))

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