"""
src/tests/tracking_pose_model_testing/test_openpifpaf.py

Smoke test for the OpenPifPaf video-annotation script.
  Script under test : src/sailsprep/tracking_pose_model_testing/openpifpaf.py
  This test file    : src/tests/tracking_pose_model_testing/test_openpifpaf.py

This script defines no functions -- it is pure top-level orchestration
(model construction + a hardcoded "/input" folder scan). There is no
business logic to unit test directly, so this file only verifies the module
executes top-to-bottom without raising once `openpifpaf` and `IPython`
(not installed here) are stubbed, `os.listdir` is patched to return no
files (so the video loop body never runs), and a couple of expected
top-level constants are present.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_openpifpaf.py -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest


@contextlib.contextmanager
def _scoped_modules(stub_map: dict):
    saved = {k: sys.modules.get(k) for k in stub_map}
    sys.modules.update(stub_map)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _stub(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def _make_stub_modules() -> dict:
    fake_net = mock.MagicMock()
    fake_net.to.return_value = fake_net
    fake_net.head_nets = mock.MagicMock()
    fake_net.base_net.stride = 8

    network_mod = _stub("openpifpaf.network", factory=mock.MagicMock(return_value=(fake_net, None)))

    keypoints_cls = mock.MagicMock()
    nms_mod = _stub("openpifpaf.decoder.nms", Keypoints=keypoints_cls)
    decoder_mod = _stub(
        "openpifpaf.decoder",
        CifSeeds=mock.MagicMock(threshold=0.5),
        nms=nms_mod,
        factory_decode=mock.MagicMock(return_value=mock.MagicMock()),
    )

    datasets_mod = _stub(
        "openpifpaf.datasets",
        PilImageList=mock.MagicMock(return_value=[]),
        collate_images_anns_meta=mock.MagicMock(),
    )

    show_mod = _stub(
        "openpifpaf.show",
        KeypointPainter=mock.MagicMock(return_value=mock.MagicMock()),
        image_canvas=mock.MagicMock(),
    )

    openpifpaf_mod = _stub(
        "openpifpaf",
        network=network_mod,
        decoder=decoder_mod,
        datasets=datasets_mod,
        show=show_mod,
    )

    ipython_display_mod = _stub("IPython.display", display=mock.MagicMock())
    ipython_mod = _stub("IPython", display=ipython_display_mod)

    # openpifpaf.py only needs torch.device(...) / torch.cuda.is_available()
    # at import time. Some other test files in this directory (e.g.
    # test_hrnet.py) permanently replace sys.modules["torch"] with a bare
    # stub without restoring it, so a minimal scoped torch stub (restored
    # afterwards via `_scoped_modules`) keeps this test correct regardless
    # of what other test files have done to sys.modules["torch"].
    torch_mod = _stub(
        "torch",
        device=mock.MagicMock(side_effect=lambda x: x),
        cuda=mock.MagicMock(is_available=mock.MagicMock(return_value=False)),
    )

    return {
        "torch": torch_mod,
        "openpifpaf": openpifpaf_mod,
        "openpifpaf.network": network_mod,
        "openpifpaf.decoder": decoder_mod,
        "openpifpaf.decoder.nms": nms_mod,
        "openpifpaf.datasets": datasets_mod,
        "openpifpaf.show": show_mod,
        "IPython": ipython_mod,
        "IPython.display": ipython_display_mod,
    }


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "openpifpaf.py"


@pytest.mark.unit
class TestOpenpifpafSmoke:

    def test_module_loads_without_raising(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("openpifpaf_pipeline", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            _scoped_modules(_make_stub_modules()),
            mock.patch("os.listdir", return_value=[]),
            mock.patch("os.makedirs"),
            mock.patch("builtins.print"),
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert mod.input_folder == "/input"
        assert mod.output_folder == "/outputs/Openpifpaf"
