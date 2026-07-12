"""
Tests for src/sailsprep/action_model_testing/pyskl/train_mlp_fusion.py

src/tests/action_model_testing/pyskl/test_train_mlp_fusion.py

"""
import pickle
import sys

import numpy as np
import pytest
import torch

from sailsprep.action_model_testing.pyskl.train_mlp_fusion import FusionMLP, main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logits_dict(model_names, n_samples, num_classes, seed):
    """Build a {model_name: (n_samples, num_classes) ndarray} dict of fake logits."""
    rng = np.random.RandomState(seed)
    return {name: rng.randn(n_samples, num_classes).astype(np.float32) for name in model_names}


def _write_fusion_pickles(work_dirs, dataset, model_names, num_classes,
                           n_val=32, n_test=16, seed=0):
    """Write fusion_{dataset}_val_logits.pkl and fusion_{dataset}_test_logits.pkl."""
    rng = np.random.RandomState(seed)

    val_labels = rng.randint(0, num_classes, size=n_val)
    test_labels = rng.randint(0, num_classes, size=n_test)

    val_data = {
        'logits': _make_logits_dict(model_names, n_val, num_classes, seed=seed + 1),
        'labels': val_labels,
    }
    test_data = {
        'logits': _make_logits_dict(model_names, n_test, num_classes, seed=seed + 2),
        'labels': test_labels,
    }

    with open(work_dirs / f'fusion_{dataset}_val_logits.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(work_dirs / f'fusion_{dataset}_test_logits.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    return val_data, test_data


# ---------------------------------------------------------------------------
# FusionMLP unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('num_classes,num_models', [(4, 2), (5, 3)])
def test_fusion_mlp_forward_shape(num_classes, num_models):
    input_dim = num_classes * num_models
    model = FusionMLP(input_dim=input_dim, num_classes=num_classes, hidden_dim=16)

    batch_size = 8
    x = torch.randn(batch_size, input_dim)
    out = model(x)

    assert out.shape == (batch_size, num_classes)


def test_fusion_mlp_eval_mode_is_deterministic():
    """Dropout should be disabled in eval mode, so repeated forward passes match."""
    model = FusionMLP(input_dim=8, num_classes=4, hidden_dim=16)
    model.eval()

    x = torch.randn(5, 8)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)

    assert torch.allclose(out1, out2)


def test_fusion_mlp_train_mode_uses_dropout():
    """In train mode, dropout makes repeated forward passes differ (with high probability)."""
    torch.manual_seed(0)
    model = FusionMLP(input_dim=32, num_classes=4, hidden_dim=64)
    model.train()

    x = torch.randn(4, 32)
    out1 = model(x)
    out2 = model(x)

    assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# main() end-to-end tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('dataset,num_classes', [('rmm', 4), ('loco', 5)])
def test_main_trains_and_saves_best_model(tmp_path, monkeypatch, dataset, num_classes):
    work_dirs = tmp_path / 'work_dirs'
    work_dirs.mkdir()

    model_names = ['stgcnpp_j', 'stgcnpp_b']
    _write_fusion_pickles(work_dirs, dataset, model_names, num_classes)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, 'argv',
        ['train_mlp_fusion.py', '--dataset', dataset, '--epochs', '2', '--hidden_dim', '8'],
    )
    # Force CPU so the test is fast and deterministic across CI runners.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    main()

    saved_model_path = work_dirs / f'fusion_mlp_{dataset}_best.pth'
    assert saved_model_path.exists()

    # Sanity check: the saved state dict loads back into a freshly built model
    # of the expected input/output dimensions.
    input_dim = num_classes * len(model_names)
    model = FusionMLP(input_dim=input_dim, num_classes=num_classes, hidden_dim=8)
    state_dict = torch.load(saved_model_path, map_location='cpu')
    model.load_state_dict(state_dict)  # raises if shapes mismatch


def test_main_reports_individual_model_accuracies(tmp_path, monkeypatch, capsys):
    dataset, num_classes = 'rmm', 4
    work_dirs = tmp_path / 'work_dirs'
    work_dirs.mkdir()

    model_names = ['stgcnpp_j', 'stgcnpp_b']
    _write_fusion_pickles(work_dirs, dataset, model_names, num_classes)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, 'argv',
        ['train_mlp_fusion.py', '--dataset', dataset, '--epochs', '2', '--hidden_dim', '8'],
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    main()

    captured = capsys.readouterr()
    assert 'Individual Model Accuracies' in captured.out
    for name in model_names:
        assert name in captured.out


def test_main_runs_weighted_average_fusion_when_four_stgcn_streams_present(
        tmp_path, monkeypatch, capsys):
    """When all four canonical STGCN++ streams are present, the script should
    also print the fixed-weight (2:2:1:1) fusion accuracy for comparison."""
    dataset, num_classes = 'rmm', 4
    work_dirs = tmp_path / 'work_dirs'
    work_dirs.mkdir()

    model_names = ['stgcnpp_j', 'stgcnpp_b', 'stgcnpp_jm', 'stgcnpp_bm']
    _write_fusion_pickles(work_dirs, dataset, model_names, num_classes)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, 'argv',
        ['train_mlp_fusion.py', '--dataset', dataset, '--epochs', '2', '--hidden_dim', '8'],
    )
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

    main()

    captured = capsys.readouterr()
    assert 'STGCN++ 4-stream weighted avg' in captured.out


def test_main_rejects_invalid_dataset_choice(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'work_dirs').mkdir()
    monkeypatch.setattr(sys, 'argv', ['train_mlp_fusion.py', '--dataset', 'not_a_real_dataset'])

    with pytest.raises(SystemExit):
        main()