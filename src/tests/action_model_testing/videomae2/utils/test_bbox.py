"""
Tests for src/sailsprep/action_model_testing/videomae2/utils/bbox.py
"""
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

_MODULE_PATH = (
    Path(__file__).parents[4]
    / "sailsprep" / "action_model_testing" / "videomae2" / "utils" / "bbox.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("videomae2_bbox", _MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load()


class TestLoadBboxMap:
    def test_parses_frame_to_bbox(self, mod, tmp_path):
        h5_path = tmp_path / "bboxes.h5"
        dtype = np.dtype([
            ("values_block_1", np.int64, (6,)),
        ])
        rows = np.zeros(2, dtype=dtype)
        rows["values_block_1"][0] = [1, 0, 10, 20, 30, 40]
        rows["values_block_1"][1] = [2, 0, 50, 60, 70, 80]

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("bboxes/table", data=rows)

        bbox_map = mod.load_bbox_map(str(h5_path))
        assert bbox_map[1] == (10, 20, 30, 40)
        assert bbox_map[2] == (50, 60, 70, 80)
