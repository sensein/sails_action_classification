import sys
import os
from pathlib import Path
# Ensure src/ is on the path so `sailsprep` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# Allow `from clips.xxx import ...` etc.
VLM_ROOT = Path(__file__).parent.parent / "sailsprep" / "action_model_testing" / "vlm_models"
sys.path.insert(0, str(VLM_ROOT))

# window_classifier_ovis/qwen do `from shared_utils import ...` (bare, same-dir style)
sys.path.insert(0, str(VLM_ROOT / "window_classification"))