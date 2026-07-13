"""
src/tests/tracking_pose_model_testing/test_deva.py

Smoke test for the DEVA (tracking-anything-with-text) video-segmentation
script.
  Script under test : src/sailsprep/tracking_pose_model_testing/deva.py
  This test file    : src/tests/tracking_pose_model_testing/test_deva.py

This script defines no functions -- it is pure top-level orchestration:
argparse setup, model construction, and a hardcoded "video_folder" scan. The
entire `deva` package (plus `groundingdino` and `IPython`) is unavailable in
this environment, so every import is stubbed. The argparse helper functions
(`add_common_eval_args`, `add_ext_eval_args`, `add_text_default_args`) are
stubbed to register just the CLI flags the script actually reads off
`args`/`cfg` after parsing (`--model` and `--num_voting_frames`), matching
the real library's contract closely enough for the script to run without
KeyError/AttributeError. `os.listdir` is patched so the hardcoded
"video_folder" scan finds no videos and the per-video loop body never runs.

Usage:
    poetry run pytest src/tests/tracking_pose_model_testing/test_deva.py -v
"""

from __future__ import annotations

import argparse
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


def _add_common_eval_args(parser: argparse.ArgumentParser) -> None:
    """Register only the flags deva.py reads off args/cfg after parsing."""
    parser.add_argument("--model", default=None)
    parser.add_argument("--num_voting_frames", default=3, type=int)


def _add_ext_eval_args(parser: argparse.ArgumentParser) -> None:
    pass


def _add_text_default_args(parser: argparse.ArgumentParser) -> None:
    pass


def _make_deva_model_cls():
    def _factory(cfg):
        model = mock.MagicMock()
        model.cuda.return_value = model
        model.eval.return_value = model
        model.load_weights = mock.MagicMock()
        return model

    return mock.MagicMock(side_effect=_factory)


def _make_inference_core_cls():
    def _factory(deva_model, config):
        core = mock.MagicMock()
        core.next_voting_frame = None
        core.enabled_long_id = mock.MagicMock()
        core.object_manager = mock.MagicMock()
        core.clear_buffer = mock.MagicMock()
        return core

    return mock.MagicMock(side_effect=_factory)


def _make_torch_stub() -> types.ModuleType:
    """
    deva.py only needs `torch.autograd.set_grad_enabled(...)` at import time
    (torch.load / torch.cuda.amp.autocast are only reached if a checkpoint is
    configured or videos are found, both of which are avoided here). Some
    other test files in this directory (e.g. test_hrnet.py) permanently
    replace sys.modules["torch"] with a bare/empty stub without restoring
    it, which can leave a real-`torch`-shaped hole depending on collection
    order. Providing our own scoped torch stub (restored via
    `_scoped_modules` after this module loads) makes this test robust
    regardless of what other test files have done to sys.modules["torch"].
    """
    autograd_mod = _stub("torch.autograd", set_grad_enabled=mock.MagicMock())
    torch_mod = _stub(
        "torch",
        autograd=autograd_mod,
        cuda=mock.MagicMock(is_available=mock.MagicMock(return_value=False)),
        load=mock.MagicMock(),
    )
    return torch_mod


def _make_stub_modules() -> dict:
    eval_args_mod = _stub(
        "deva.inference.eval_args",
        add_common_eval_args=_add_common_eval_args,
        get_model_and_config=mock.MagicMock(),
    )
    ext_eval_args_mod = _stub(
        "deva.ext.ext_eval_args",
        add_ext_eval_args=_add_ext_eval_args,
        add_text_default_args=_add_text_default_args,
    )
    grounding_dino_mod = _stub(
        "deva.ext.grounding_dino",
        get_grounding_dino_model=mock.MagicMock(
            return_value=(mock.MagicMock(), mock.MagicMock())
        ),
    )
    with_text_processor_mod = _stub(
        "deva.ext.with_text_processor",
        process_frame_with_text=mock.MagicMock(),
    )
    demo_utils_mod = _stub("deva.inference.demo_utils", flush_buffer=mock.MagicMock())
    inference_core_mod = _stub(
        "deva.inference.inference_core", DEVAInferenceCore=_make_inference_core_cls()
    )
    result_utils_mod = _stub(
        "deva.inference.result_utils", ResultSaver=mock.MagicMock(return_value=mock.MagicMock())
    )
    network_mod = _stub("deva.model.network", DEVA=_make_deva_model_cls())

    deva_ext_pkg = _stub("deva.ext")
    deva_inference_pkg = _stub("deva.inference")
    deva_model_pkg = _stub("deva.model")
    deva_pkg = _stub("deva")

    groundingdino_inference_mod = _stub(
        "groundingdino.util.inference", Model=mock.MagicMock()
    )
    groundingdino_util_pkg = _stub("groundingdino.util")
    groundingdino_pkg = _stub("groundingdino")

    ipython_display_mod = _stub("IPython.display", HTML=mock.MagicMock())
    ipython_mod = _stub("IPython", display=ipython_display_mod)

    return {
        "torch": _make_torch_stub(),
        "deva": deva_pkg,
        "deva.ext": deva_ext_pkg,
        "deva.ext.ext_eval_args": ext_eval_args_mod,
        "deva.ext.grounding_dino": grounding_dino_mod,
        "deva.ext.with_text_processor": with_text_processor_mod,
        "deva.inference": deva_inference_pkg,
        "deva.inference.demo_utils": demo_utils_mod,
        "deva.inference.eval_args": eval_args_mod,
        "deva.inference.inference_core": inference_core_mod,
        "deva.inference.result_utils": result_utils_mod,
        "deva.model": deva_model_pkg,
        "deva.model.network": network_mod,
        "groundingdino": groundingdino_pkg,
        "groundingdino.util": groundingdino_util_pkg,
        "groundingdino.util.inference": groundingdino_inference_mod,
        "IPython": ipython_mod,
        "IPython.display": ipython_display_mod,
    }


def _find_src_root(start: Path) -> Path:
    for parent in start.parents:
        if parent.name == "src":
            return parent
    raise RuntimeError(f"Could not locate 'src' directory above {start}")


_SRC_ROOT = _find_src_root(Path(__file__))
PIPELINE_SCRIPT = _SRC_ROOT / "sailsprep" / "tracking_pose_model_testing" / "deva.py"


@pytest.mark.unit
class TestDevaSmoke:

    def test_module_loads_without_raising(self):
        if not PIPELINE_SCRIPT.exists():
            pytest.skip(f"Pipeline script not found: {PIPELINE_SCRIPT}")

        spec = importlib.util.spec_from_file_location("deva_pipeline", PIPELINE_SCRIPT)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]

        with (
            _scoped_modules(_make_stub_modules()),
            mock.patch("os.listdir", return_value=[]),
            mock.patch("os.makedirs"),
            mock.patch("builtins.print"),
        ):
            spec.loader.exec_module(mod)  # type: ignore[union-attr]

        assert mod.CLASSES == ["person"]
        assert mod.cfg["prompt"] == "person"
        assert mod.SOURCE_VIDEO_DIR == "video_folder"
        assert mod.cfg["enable_long_term"] is True
        assert mod.cfg["max_num_objects"] == 50
