"""
Tests for src/sailsprep/action_model_testing/Videomaev2/utils/collate.py
"""
import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "Videomaev2" / "utils" / "collate.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("videomae2_collate", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


class TestCollate:
    def test_stacks_videos_and_labels(self, mod):
        batch = [
            (torch.zeros(3, 8, 224, 224), 0),
            (torch.ones(3, 8, 224, 224), 1),
        ]
        videos, labels = mod.collate(batch)
        assert videos.shape == (2, 3, 8, 224, 224)
        assert labels.tolist() == [0, 1]
        assert labels.dtype == torch.long
