"""Tests for the training machinery (scripts/base_training.py, scripts/experiments/ablation_study.py).

Synthetic, CPU-only tests of the configuration defaults, the physics gate, the training
pass and the metric helpers, plus strict-xfail targets for the agreed changes: the fixed
train/validation split (Phase 2), the model built from the configuration with its Stokes
channels (Phase 1c) and the MDI flavor (Phase 1e).
"""
import contextlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from models.pinn_mscnn_model import PhysicsInformedMSCNN
from scripts.base_training import (
    BalancedStepTensorDataset,
    MetricsLogger,
    TrainingConfig,
    choose_balanced_cache_runtime_mode,
    initialize_wfa_gate_state,
    train_epoch,
    train_one_step,
    update_wfa_gate_state,
)
from utils.normalizer import MhdNormalizer

ROOT = Path(__file__).resolve().parent.parent


def _cfg(tmp_path, **kwargs) -> TrainingConfig:
    """Configuration writing its checkpoints and logs under tmp_path, on the CPU."""
    return TrainingConfig(checkpoint_dir=str(tmp_path / "ck"), log_dir=str(tmp_path / "logs"), device="cpu",
                          **kwargs)


# --- Configuration -----------------------------------------------------------------------

def test_nicole_source_uses_its_own_statistics_caches_and_continuum_mode(tmp_path):
    """The nicole_tau500 source never reads or overwrites the legacy files.

    Its normalization statistics and caches get ``nicole_tau500`` paths, and the continuum
    mode is ``per_step`` because the profiles are already normalized to the HSRA
    continuum (a global legacy I_c would rescale them wrongly).
    """
    cfg = _cfg(tmp_path)
    assert cfg.data_source == "nicole_tau500"
    assert cfg.mhd_normalizer_path == "normalization_stats/nicole_tau500/mhd_normalization.json"
    assert cfg.stokes_normalizer_path == "normalization_stats/nicole_tau500/stokes_normalization.json"
    assert cfg.cache_dir.endswith("_nicole_tau500") and cfg.balanced_cache_dir.endswith("_nicole_tau500")
    assert cfg.stokes_ic_mode == "per_step"


def test_logtau_grid_matches_the_tau500_files_exactly(tmp_path):
    """The default grid is the 45 nodes -3.0 to 1.4 that the tau_500 files were generated on.

    The loader refuses a file whose grid differs by more than 1e-6, so the grid must be
    built in float64 before the cast (a float32 arange drifts by about 4e-6).
    """
    g = _cfg(tmp_path).get_logtau_values()
    reference = np.round(np.arange(-3.0, 1.45, 0.1).astype(np.float32), 6)
    assert g.size == 45
    np.testing.assert_array_equal(g, reference)


def test_config_round_trips_through_json(tmp_path):
    """A saved configuration reloads with the same physics settings."""
    cfg = _cfg(tmp_path, lambda_wfa=10.0, blos_physics_mode="single_height", blos_target_logtau=-0.8)
    cfg.save(tmp_path / "c.json")
    back = TrainingConfig.load(tmp_path / "c.json")
    assert (back.lambda_wfa, back.blos_physics_mode, back.blos_target_logtau) == (10.0, "single_height", -0.8)


def test_balanced_cache_preloads_only_within_the_ram_budget(tmp_path):
    """``auto`` preloads when the estimate fits in budget x fraction (32 GB x 0.75 = 24 GB), else reads from disk."""
    cfg = _cfg(tmp_path)
    assert choose_balanced_cache_runtime_mode(cfg, 23 * 1024 ** 3) == "preload"
    assert choose_balanced_cache_runtime_mode(cfg, 25 * 1024 ** 3) == "disk"
    assert choose_balanced_cache_runtime_mode(_cfg(tmp_path, balanced_cache_strategy="disk"), 1) == "disk"


# --- Physics gate ------------------------------------------------------------------------

@pytest.mark.parametrize("lambdas", [dict(lambda_wfa=1.0), dict(lambda_doppler=0.5), dict(lambda_temp=2.0)])
def test_gate_starts_closed_for_every_physics_flavor(tmp_path, lambdas):
    """With the plateau gate, every flavor's physics term starts off, not only the WFA's.

    Regression: the gate used to open whenever lambda_wfa was 0, so doppler_only and
    black_body_only had their term on from epoch 1 while wfa_only waited.
    """
    base = dict(lambda_wfa=0.0, lambda_doppler=0.0, lambda_temp=0.0)
    base.update(lambdas)
    cfg = _cfg(tmp_path, wfa_gate_mode="plateau", **base)
    assert initialize_wfa_gate_state(cfg)["enabled"] is False


def test_gate_is_open_for_the_baseline_and_when_off(tmp_path):
    """The baseline has nothing to gate, and mode ``off`` means always on."""
    assert initialize_wfa_gate_state(_cfg(tmp_path, wfa_gate_mode="plateau", lambda_wfa=0.0, lambda_doppler=0.0,
                                          lambda_temp=0.0))["enabled"]
    assert initialize_wfa_gate_state(_cfg(tmp_path, wfa_gate_mode="off", lambda_wfa=1.0))["enabled"]


def test_plateau_gate_opens_after_patience_epochs_without_improvement(tmp_path):
    """The gate opens once ``patience`` epochs in a row fail to beat the best MSE by more than ``min_delta``.

    With the paper's values (patience 5, min_delta 5e-4), a flat MSE after a descent opens
    the gate on the fifth flat epoch, and it stays open afterwards.
    """
    cfg = _cfg(tmp_path, wfa_gate_mode="plateau", wfa_gate_patience=5, wfa_gate_min_delta=5e-4, lambda_wfa=1.0)
    state = initialize_wfa_gate_state(cfg)
    mse = [1.0, 0.5, 0.3] + [0.3] * 7
    opened = [update_wfa_gate_state(state, cfg, e, m)[1] for e, m in enumerate(mse)]
    assert opened.index(True) == 7 and opened.count(True) == 1
    assert state["enabled"] and state["trigger_epoch"] == 8


def test_plateau_is_measured_against_the_best_mse_not_the_previous_epoch(tmp_path):
    """A slow steady decline keeps resetting the plateau counter, so the gate never opens.

    Improvements are measured from the best MSE so far. Falling by 1e-4 per epoch adds
    up to more than 5e-4 within five epochs, which counts as an improvement and resets the
    count: with patience 5 and min_delta 5e-4 such a decline never opens the gate.
    """
    cfg = _cfg(tmp_path, wfa_gate_mode="plateau", wfa_gate_patience=5, wfa_gate_min_delta=5e-4, lambda_wfa=1.0)
    state = initialize_wfa_gate_state(cfg)
    opened = [update_wfa_gate_state(state, cfg, e, 0.3 - 1.01e-4 * e)[1] for e in range(40)]
    assert not any(opened)


def test_gate_waits_for_the_warmup_epochs(tmp_path):
    """No opening during the warm-up, even if the threshold is already met."""
    cfg = _cfg(tmp_path, wfa_gate_mode="threshold", wfa_gate_threshold=0.5, wfa_gate_warmup_epochs=3, lambda_wfa=1.0)
    state = initialize_wfa_gate_state(cfg)
    opened = [update_wfa_gate_state(state, cfg, e, 0.1)[1] for e in range(5)]
    assert opened == [False, False, False, True, False]


# --- Training pass -----------------------------------------------------------------------

def _tiny_step(tmp_path, n=64):
    """A normalizer, a small balanced dataset of one step and its approximation maps."""
    rng = np.random.default_rng(0)
    atm = {"T": 5000 + 300 * rng.standard_normal((8, 8, 45)), "Vz": rng.standard_normal((8, 8, 45)),
           "Bz": 100 * rng.standard_normal((8, 8, 45))}
    norm = MhdNormalizer(n_tau=45, raw_values_dir=str(tmp_path))
    norm.update(atm)
    with contextlib.redirect_stdout(io.StringIO()):
        norm.finalize()
    ds = BalancedStepTensorDataset(rng.standard_normal((n, 2, 112)).astype(np.float32),
                                   rng.standard_normal((n, 135)).astype(np.float32),
                                   rng.integers(0, 8, (n, 2)))
    approx = {"blos": np.full((8, 8), 100.0), "vlos": np.full((8, 8), 1.0), "temp": np.full((8, 8), 6000.0)}
    return norm, ds, approx


def _model(**lambdas) -> PhysicsInformedMSCNN:
    """Small model on the 45-node grid with single-height physics terms."""
    base = dict(lambda_wfa=0.0, lambda_doppler=0.0, lambda_temp=0.0)
    base.update(lambdas)
    return PhysicsInformedMSCNN(scales=[1], output_features=135, blos_physics_mode="single_height",
                                blos_target_logtau=-0.8, vlos_physics_mode="single_height", vlos_target_logtau=-1.0,
                                **base)


def test_one_step_updates_the_weights_and_marks_a_closed_gate_as_nan(tmp_path):
    """A pass changes the weights; with the gate closed the physics losses are logged as NaN, not 0.

    NaN keeps the CSV honest: a 0 would read as "the constraint is perfectly met".
    """
    norm, ds, approx = _tiny_step(tmp_path)
    cfg = _cfg(tmp_path, batch_size=16)
    torch.manual_seed(0)
    m = _model(lambda_wfa=1.0)
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    out = train_one_step(m, DataLoader(ds, batch_size=16), approx, norm, opt, cfg, 0, 110, None,
                         enable_physics=False)
    assert any(not torch.equal(a, b) for a, b in zip(before, m.parameters()))
    assert np.isfinite(out["mse_loss"]) and np.isnan(out["wfa_loss"]) and np.isnan(out["physics_loss"])


def test_epoch_visits_every_training_step_once(tmp_path):
    """One epoch uses every training step exactly once and counts every sample.

    Each step's samples come from the preloaded balanced cache here; the epoch's losses
    are averages over steps.
    """
    norm, ds, approx = _tiny_step(tmp_path)
    cfg = _cfg(tmp_path, batch_size=32)
    m = _model()
    logger = MetricsLogger(tmp_path / "logs")
    preloaded = {s: (ds, approx) for s in (110, 120, 130, 140)}
    with contextlib.redirect_stderr(io.StringIO()):
        out = train_epoch(m, [110, 120, 130, 140], cfg, norm, None, torch.optim.Adam(m.parameters()), 0, logger,
                          preloaded_balanced_steps=preloaded)
    logger.close()
    rows = (tmp_path / "logs" / "batch_log.csv").read_text().splitlines()[1:]
    assert out["n_steps"] == 4 and out["n_pixels_used"] == 4 * len(ds)
    assert sorted(int(r.split(",")[1]) for r in rows) == [110, 120, 130, 140]


def test_training_histograms_use_the_exact_inverse_of_the_normalization(tmp_path):
    """The histogram helper's inverse returns T, Vz and Bz exactly, Vz included.

    Unlike ``MhdNormalizer.denormalize`` (strict xfail for Vz), it does not add the mean
    back for the "centered" Vz, which is the exact inverse of the transform.
    """
    from scripts.experiments.ablation_study import _denormalize_targets_from_stats
    rng = np.random.default_rng(1)
    atm = {"T": 5000 + 300 * rng.standard_normal((6, 6, 3)), "Vz": 0.5 + rng.standard_normal((6, 6, 3)),
           "Bz": 100 * rng.standard_normal((6, 6, 3))}
    norm = MhdNormalizer(n_tau=3, raw_values_dir=str(tmp_path))
    norm.update(atm)
    with contextlib.redirect_stdout(io.StringIO()):
        norm.finalize()
    z = norm.transform(atm)
    for p in ("T", "Vz", "Bz"):
        back = _denormalize_targets_from_stats(z[p].reshape(-1, 3), p, norm)
        np.testing.assert_allclose(back, atm[p].reshape(-1, 3), rtol=1e-4, atol=1e-2)


# --- Targets for the agreed changes ------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 2): training on steps 110-140 with step 150 "
                   "fixed for validation. Today validation is a random 10 % of the training steps.")
def test_validation_step_is_fixed_and_excluded_from_training(tmp_path):
    """``resolve_step_split`` honours explicit validation steps and keeps them out of training.

    With steps 110 to 150 every 10, test steps 198 to 200 and ``val_steps=[150]``, the
    split must be train 110, 120, 130, 140 and validation 150, whatever the seed.
    """
    from scripts.base_training import resolve_step_split
    cfg = _cfg(tmp_path, val_steps=[150])
    for seed in (0, 42):
        train, val = resolve_step_split(list(range(110, 151, 10)), test_steps=[198, 199, 200],
                                        val_steps=cfg.val_steps, seed=seed)
        assert (train, val) == ([110, 120, 130, 140], [150])


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1c): the model is built from the "
                   "configuration, with as many input channels as Stokes channels (I, Q, U, V by default).")
def test_model_is_built_from_the_configuration(tmp_path):
    """``build_model(config)`` uses the configured architecture and one input channel per Stokes channel.

    ``ablation_study.run_single_experiment`` hardcodes 2 channels today, so a 4-channel
    configuration would silently train a 2-channel model.
    """
    from scripts.base_training import build_model
    model = build_model(_cfg(tmp_path, c1_filters=8, lambda_wfa=10.0))
    assert next(model.msfm.conv_block.c1.parameters()).shape[:2] == (8, 4)
    assert model.lambda_wfa == 10.0
    assert model(torch.zeros(2, 4, 112)).shape == (2, 135)


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1e): an mdi_only flavor with its own "
                   "lambda and comparison node.")
def test_ablation_offers_the_mdi_flavor():
    """The ablation CLI accepts ``--experiments mdi_only``, ``--lambda_mdi`` and ``--mdi_target_logtau``."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "experiments" / "ablation_study.py"), "--help"],
                         capture_output=True, text=True, timeout=120)
    assert "mdi_only" in out.stdout and "--lambda_mdi" in out.stdout and "--mdi_target_logtau" in out.stdout


class _EchoModel(torch.nn.Module):
    """Stand-in model whose prediction for each sample is a stored normalized target vector."""

    def __init__(self, outputs: torch.Tensor):
        """Store the outputs to return, in sample order."""
        super().__init__()
        self.outputs = outputs
        self.pos = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the next ``len(x)`` stored outputs."""
        out = self.outputs[self.pos:self.pos + len(x)]
        self.pos += len(x)
        return out

    def set_physics_context(self, *args, **kwargs):
        """Accept and ignore the physics context."""


@pytest.mark.xfail(strict=True, reason="Agreed with the user (2026-11-23, Phase 1): the per-epoch MURaM test metrics "
                   "compare predictions with MURaM's atmosphere at the paper nodes, not with the physics approximations.")
def test_muram_test_metrics_compare_with_the_ground_truth(tmp_path, monkeypatch):
    """``compute_muram_test_metrics`` scores predictions against MURaM's T, Vz and Bz at each requested node.

    The stand-in model predicts exactly the normalized targets, and the approximation
    maps are deliberately unrelated (constants). Against the ground truth the correlation
    must be 1 and the RRMSE 0 at every node; a function scoring against the
    approximations would fail.
    """
    import scripts.experiments.ablation_study as ablation_study
    norm, ds, _ = _tiny_step(tmp_path, n=60)
    approx = {"blos": np.zeros((8, 8)), "vlos": np.zeros((8, 8)), "temp": np.full((8, 8), 6000.0)}
    monkeypatch.setattr(ablation_study, "load_and_prepare_step", lambda **kw: (ds, approx))
    model = _EchoModel(torch.from_numpy(ds.mhd_targets))
    cfg = _cfg(tmp_path)
    metrics = ablation_study.compute_muram_test_metrics(model, [198], cfg, norm, None, nodes=(-2.0, -0.8, 0.0))
    for q in ("T", "Vz", "Bz"):
        for node in (-2.0, -0.8, 0.0):
            assert metrics[q][node]["correlation"] == pytest.approx(1.0, abs=1e-6)
            assert metrics[q][node]["rrmse"] == pytest.approx(0.0, abs=1e-6)
