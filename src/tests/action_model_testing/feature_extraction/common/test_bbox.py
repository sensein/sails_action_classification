"""
Tests for src/sailsprep/action_model_testing/feature_extraction/common/bbox.py
"""
import h5py
import numpy as np

from sailsprep.action_model_testing.feature_extraction.common.bbox import (
    crop_frame_with_bbox,
    load_bbox_map,
)


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


class TestCropFrameWithBbox:
    def test_output_shape(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = crop_frame_with_bbox(frame, (10, 10, 60, 60), out_size=224)
        assert crop.shape == (224, 224, 3)

    def test_clamps_bbox_to_frame_bounds(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        crop = crop_frame_with_bbox(frame, (-10, -10, 200, 200), out_size=64)
        assert crop.shape == (64, 64, 3)

    def test_degenerate_bbox_still_produces_output(self):
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        crop = crop_frame_with_bbox(frame, (10, 10, 10, 10), out_size=32)
        assert crop.shape == (32, 32, 3)
