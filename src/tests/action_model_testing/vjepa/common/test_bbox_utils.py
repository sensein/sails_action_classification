"""
Tests for src/sailsprep/action_model_testing/vjepa/common/bbox_utils.py
"""
import h5py
import numpy as np

from sailsprep.action_model_testing.vjepa.common.bbox_utils import load_bbox_map


class TestLoadBboxMap:
    def test_parses_frame_to_bbox(self, tmp_path):
        h5_path = tmp_path / "bboxes.h5"
        dtype = np.dtype([("values_block_1", np.int64, (6,))])
        rows = np.zeros(2, dtype=dtype)
        rows["values_block_1"][0] = [1, 0, 10, 20, 30, 40]
        rows["values_block_1"][1] = [2, 0, 50, 60, 70, 80]

        with h5py.File(h5_path, "w") as f:
            f.create_dataset("bboxes/table", data=rows)

        bbox_map = load_bbox_map(str(h5_path))
        assert bbox_map[1] == (10, 20, 30, 40)
        assert bbox_map[2] == (50, 60, 70, 80)
