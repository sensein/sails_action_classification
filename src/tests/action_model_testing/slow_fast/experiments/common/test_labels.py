"""
Tests for src/sailsprep/action_model_testing/slow_fast/experiments/common/labels.py
"""
from sailsprep.action_model_testing.slow_fast.experiments.common.labels import (
    ACTION_CLASSES,
    CLASS_TO_IDX,
    CSV_CLASS_TO_INTERNAL,
    IDX_TO_CLASS,
)


class TestLabelMaps:
    def test_action_classes_match_csv_mapping_values(self):
        assert set(ACTION_CLASSES) == set(CSV_CLASS_TO_INTERNAL.values())

    def test_class_to_idx_is_bijective_with_idx_to_class(self):
        for cls, idx in CLASS_TO_IDX.items():
            assert IDX_TO_CLASS[idx] == cls

    def test_indices_are_contiguous(self):
        assert sorted(CLASS_TO_IDX.values()) == list(range(len(ACTION_CLASSES)))
