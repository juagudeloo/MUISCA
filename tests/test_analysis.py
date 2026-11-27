"""Tests for utils/analysis.py: metrics, labels, MODEST geometry and multi-model figures.

The metrics feed the paper's tables, so they are checked against direct numpy formulas.
Strict-xfail targets cover the agreed changes: four-channel checkpoints (Phase 1c), the
MDI flavor's name and order (Phase 1e) and five-model comparison figures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from utils import analysis
from utils.analysis import (
    AnalysisModelPipeline,
    compute_distribution_similarity_metrics,
    compute_metric_vs_tau,
    compute_metrics_by_field_strength,
    compute_regression_metrics,
    modest_crop_extent,
    ordered_models,
    param_axis_label,
)


# --- Metrics -----------------------------------------------------------------------------

def test_regression_metrics_match_their_definitions():
    """RMSE, RRMSE = RMSE / mean(|truth|), MAE, bias = mean(pred - truth), Pearson r and R^2 equal numpy's.

    NaN pairs are dropped first. These are the numbers of the paper's tables.
    """
    rng = np.random.default_rng(0)
    t = 1000 + 300 * rng.standard_normal(500)
    p = t + 50 * rng.standard_normal(500) + 20
    t_nan, p_nan = np.append(t, np.nan), np.append(p, 1.0)
    m = compute_regression_metrics(t_nan, p_nan)
    rmse = np.sqrt(np.mean((p - t) ** 2))
    assert m["n_points"] == 500
    assert m["rmse"] == pytest.approx(rmse)
    assert m["rrmse"] == pytest.approx(rmse / np.mean(np.abs(t)))
    assert m["bias"] == pytest.approx(np.mean(p - t))
    assert m["corr"] == pytest.approx(np.corrcoef(t, p)[0, 1])
    assert m["r2"] == pytest.approx(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))


def test_distribution_metrics_are_zero_for_identical_and_maximal_for_disjoint_samples():
    """KS 0 and overlap 1 for identical samples; KS 1 and overlap 0 for samples with no common range."""
    rng = np.random.default_rng(1)
    a = rng.standard_normal(4000)
    same = compute_distribution_similarity_metrics(a, a)
    assert same["ks"] == 0.0 and same["overlap"] == pytest.approx(1.0) and same["jsd"] == pytest.approx(0.0, abs=1e-9)
    apart = compute_distribution_similarity_metrics(a, a + 100.0)
    assert apart["ks"] == 1.0 and apart["overlap"] == pytest.approx(0.0)
    assert apart["w1_quantile"] == pytest.approx(100.0, rel=1e-6)


def test_field_strength_bins_use_the_reference_field_not_the_prediction():
    """Pixels are binned by the reference |B| (G), so the split does not depend on the model.

    A prediction wildly off in a weak-field pixel must still be counted in the weak bin.
    """
    truth = np.array([10.0, 20.0, 1200.0])
    pred = np.array([5000.0, 20.0, 1100.0])
    rows = compute_metrics_by_field_strength(truth, pred, field_strength=truth)
    counts = {r["field_bin"]: r["n_points"] for r in rows}
    assert counts["0-100G"] == 2 and counts["1000-1500G"] == 1 and counts[">1500G"] == 0


def test_metric_vs_tau_sorts_nodes_and_skips_mismatched_grids():
    """Nodes come back sorted; a node whose maps differ in shape gives NaN instead of an error."""
    a = np.arange(1.0, 5.0).reshape(2, 2)
    out = compute_metric_vs_tau([(0.0, a, a), (-2.0, a, 2 * a), (-0.8, a, a[:1])])
    assert out["tau"].tolist() == [-2.0, -0.8, 0.0]
    assert out["corr"][0] == pytest.approx(1.0) and np.isnan(out["rrmse"][1]) and out["rrmse"][2] == 0.0


# --- Labels and geometry -----------------------------------------------------------------

def test_every_axis_label_carries_its_unit():
    """T is labelled in K, v in km/s and B in G, whatever the prefix."""
    assert param_axis_label("T").endswith("[K]")
    assert "km" in param_axis_label("Vz", "Pred - GT", los=True)
    assert param_axis_label("Bz", los=True) == r"$B_{\mathrm{LOS}}$ [G]"


def test_modest_extent_puts_solar_x_on_the_horizontal_axis():
    """The whole-scene extent spans nx x 0.32 arcsec horizontally (Solar-X) and ny x 0.32 vertically.

    Regression for the transposed MODEST maps: a crop of 858 columns by 390 rows must be
    274.56 arcsec wide and 124.8 arcsec tall, centred on the scan's FOV centre.
    """
    ext, labels = modest_crop_extent((0, 390, 0, 858), (390, 858))
    assert labels == ("Solar-X [arcsec]", "Solar-Y [arcsec]")
    assert ext[1] - ext[0] == pytest.approx(858 * 0.32)
    assert ext[3] - ext[2] == pytest.approx(390 * 0.32)
    assert (ext[0] + ext[1]) / 2 == pytest.approx(16.0 - 0.16) and (ext[2] + ext[3]) / 2 == pytest.approx(-126.0 - 0.16)


def test_models_follow_the_canonical_order():
    """Comparison figures list the baseline first, then the flavors in a fixed order, then any extras."""
    assert ordered_models({"black_body_only": 0, "x": 0, "no_physics": 0, "wfa_only": 0}) == \
        ["no_physics", "wfa_only", "black_body_only", "x"]


def test_short_flavor_names_select_their_models():
    """``wfa`` selects ``wfa_only`` and its lambda variants; an unknown name raises."""
    cfgs = {k: {"experiment_key": k} for k in ("no_physics", "wfa_only", "wfa_only-lambda-10", "doppler_only")}
    assert sorted(AnalysisModelPipeline.select_model_configs(cfgs, ["wfa"])) == ["wfa_only", "wfa_only-lambda-10"]
    with pytest.raises(ValueError):
        AnalysisModelPipeline.select_model_configs(cfgs, ["mdi_only_typo"])


def test_common_nodes_are_matched_between_spinor_and_the_model():
    """SPINOR's nodes -2.0, -0.8 and 0.0 are found in the 45-node grid, with their indices."""
    grid = [round(v, 6) for v in np.round(np.arange(-3.0, 1.45, 0.1), 6)]
    matches = AnalysisModelPipeline.common_tau_matches([-2.0, -0.8, 0.0], grid)
    assert [(t, i_pred) for t, _, i_pred in matches] == [(-2.0, 10), (-0.8, 22), (0.0, 30)]


def test_runtime_config_replays_the_degradation_a_run_was_trained_with(tmp_path):
    """Analysis rebuilds a run's PSF and noise settings, and treats a run without them as pre-Phase-1b.

    The defaults now blur the profiles with the Hinode PSF. A model trained before that
    (no ``spatial_psf`` in its experiment_config.json) must be evaluated on unblurred
    profiles, or every MURaM metric of the old models would change silently.
    """
    import json
    pipe = AnalysisModelPipeline("cpu", tmp_path)
    base = {"lambda_wfa": 0.0, "lambda_doppler": 0.0, "lambda_temp": 0.0}
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"data_config": {"seed": 1}}))
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"data_config": {"spatial_psf": True, "hinode_noise": 1.2e-3,
                                               "psf_path": "hinode-MODEST/PSFs/x.fits"}}))
    cfg_old = pipe.build_runtime_training_config({**base, "config_path": str(old)})
    cfg_none = pipe.build_runtime_training_config({**base, "config_path": None})
    cfg_new = pipe.build_runtime_training_config({**base, "config_path": str(new)})
    assert (cfg_old.spatial_psf, cfg_old.hinode_noise) == (False, 0.0)
    assert (cfg_none.spatial_psf, cfg_none.hinode_noise) == (False, 0.0)
    assert (cfg_new.spatial_psf, cfg_new.hinode_noise, cfg_new.psf_path) == (True, 1.2e-3, "hinode-MODEST/PSFs/x.fits")


# --- Targets for the agreed changes ------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1c): checkpoints with I, Q, U, V inputs "
                   "load in the analysis. Today AnalysisModelPipeline.load_model hardcodes 2 input channels.")
def test_pipeline_loads_a_four_channel_checkpoint(tmp_path):
    """``load_model`` reads the number of input channels from the checkpoint (here 4) instead of assuming 2."""
    from models.pinn_mscnn_model import PhysicsInformedMSCNN
    model = PhysicsInformedMSCNN(scales=[1, 2, 3], in_channels=4, output_features=135)
    path = tmp_path / "final_model.pth"
    torch.save({"model_state_dict": model.state_dict()}, path)
    pipe = AnalysisModelPipeline("cpu", tmp_path)
    loaded, n_tau = pipe.load_model({"path": path, "lambda_wfa": 0.0, "lambda_doppler": 0.0, "lambda_temp": 0.0})
    assert n_tau == 45 and loaded(torch.zeros(1, 4, 112)).shape == (1, 135)


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1e): the MDI flavor appears in every "
                   "figure and table with its own name, after the Doppler flavor.")
def test_mdi_flavor_has_a_display_name_and_a_place_in_the_order():
    """``mdi_only`` is shown as "MDI" and sits right after ``doppler_only`` in the canonical order."""
    assert analysis.display_name("mdi_only") == "MDI"
    order = list(analysis.MODEL_CANONICAL_ORDER)
    assert order.index("mdi_only") == order.index("doppler_only") + 1


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan: five flavors (baseline, WFA, Doppler, Black "
                   "Body, MDI). The combined scatter grid is a fixed 2 x 2 and drops the fifth model.")
def test_combined_scatter_grid_shows_every_model(tmp_path, monkeypatch):
    """With five models, the combined scatter figure has one titled panel per model."""
    captured = {}
    monkeypatch.setattr(plt, "close", lambda fig=None: captured.setdefault("fig", fig))
    rng = np.random.default_rng(0)
    truth = rng.standard_normal((20, 20))
    names = ["no_physics", "wfa_only", "doppler_only", "black_body_only", "mdi_only"]
    preds = {m: truth + 0.1 * rng.standard_normal((20, 20)) for m in names}
    analysis.plot_combined_jointplot_grid(truth, preds, "Bz", "test", -0.8, tmp_path / "g.png",
                                          model_order=tuple(names))
    titles = [ax.get_title() for ax in captured["fig"].axes if ax.get_title()]
    assert len(titles) == 5
