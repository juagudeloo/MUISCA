"""Tests for the scripts that build the paper's numbers and the training atmospheres.

Covers the paper-table logic (stars, paired bootstrap of the chi^2 medians), the tau_500
integration used to build the training atmospheres, and the fine-tuning configuration
guard, plus a strict-xfail target for the five-model tables on the new validation step.
"""
import json
import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
import scripts.analysis.make_paper_tables as tables  # noqa: E402


# --- Paper tables ------------------------------------------------------------------------

def test_modest_table_stars_the_highest_correlation_and_lowest_rrmse_of_each_row(monkeypatch):
    """In each row, the star goes to the largest correlation and to the smallest RRMSE (unrounded values).

    Two models are 0.0004 apart in correlation, which rounds to the same three decimals;
    the star must still go to the larger unrounded value.
    """
    monkeypatch.setattr(tables, "REGIONS", ["plage"])
    monkeypatch.setattr(tables, "LOGTAUS", [-0.8])
    monkeypatch.setattr(tables, "MODELS", ["a", "b", "c"])
    monkeypatch.setattr(tables, "MODEL_LABELS", ["A", "B", "C"])
    metrics = {("plage", "a", "Bz", -0.8): (0.5001, 2.0), ("plage", "b", "Bz", -0.8): (0.5005, 1.5),
               ("plage", "c", "Bz", -0.8): (0.30, 3.0)}
    row = [l for l in tables.modest_table(metrics, "Bz", "tab:x", "cap").splitlines() if l.startswith("Plage")][0]
    cells = row.rstrip().removesuffix("\\\\").split("&")[2:]
    assert [c.strip() for c in cells] == ["$0.500$", "$0.500^\\star$", "$0.300$", "$2.000$", "$1.500^\\star$", "$3.000$"]


def test_chi2_bootstrap_is_paired_and_reproducible():
    """The bootstrap uses the same resampled pixels for every model, and a fixed seed gives the same table.

    Model B is model A plus 1 on every pixel, so in every paired resample B's median is
    above A's: the rank must hold in all resamples (p_rank = 1) even though the two
    distributions overlap widely.
    """
    rng_data = np.random.default_rng(0)
    a = rng_data.lognormal(0.0, 1.0, 300)
    chi2 = np.column_stack([a, a + 1.0])
    s1 = tables.chi2_stats(chi2, np.random.default_rng(tables.BOOTSTRAP_SEED))
    s2 = tables.chi2_stats(chi2, np.random.default_rng(tables.BOOTSTRAP_SEED))
    assert s1["best"] == 0 and s1["p_rank"] == 1.0
    np.testing.assert_array_equal(s1["lo"], s2["lo"])
    assert (s1["lo"] <= s1["median"]).all() and (s1["median"] <= s1["hi"]).all()


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phases 2 and 3): five flavors including MDI, "
                   "and step 150 as the held-out step for the approximation table.")
def test_paper_tables_cover_five_models_on_the_new_validation_step():
    """The tables list baseline, WFA, Doppler, MDI and Black Body, and the approximation table uses step 150."""
    assert tables.MODELS == ["no_physics", "wfa_only", "doppler_only", "mdi_only", "black_body_only"]
    assert tables.APPROX_STEP == 150


# --- tau_500 scale of the training atmospheres -------------------------------------------

class _Column:
    """Minimal stand-in for MhdData holding one column of constant T, P and rho."""

    def __init__(self, nz: int, T: float, P: float, rho: float):
        """Store ``data`` as (1, 1, nz) Quantities in K, dyn cm^-2 and g cm^-3."""
        shape = (1, 1, nz)
        self.data = {"T": np.full(shape, T) * u.K, "P": np.full(shape, P) * u.dyn / u.cm ** 2,
                     "rho": np.full(shape, rho) * u.g / u.cm ** 3}


def test_tau500_is_integrated_from_the_top_of_the_box_down():
    """With uniform gas, tau_500 grows linearly downward from 0 at the top layer (z index up).

    Each 10 km layer adds kappa_500 rho dz, so tau at layer z is that increment times
    (nz - 1 - z); the top layer is clipped to log tau = -30. An integration in the wrong
    direction would put the photosphere at the top of the box.
    """
    from scripts.synthesis.tau500_multi_step_regen import compute_logtau500
    from utils.tau500_opacity import kappa_500
    nz, T, P, rho = 6, 6000.0, 1e5, 2e-7
    logtau = compute_logtau500(_Column(nz, T, P, rho), dz_km=10.0)[0, 0]
    per_layer = float(kappa_500(np.array([T]), np.array([P]), np.array([rho]))[0]) * rho * 1e6
    np.testing.assert_allclose(10 ** logtau[:-1], per_layer * np.arange(nz - 1, 0, -1), rtol=1e-9)
    assert logtau[-1] == pytest.approx(-30.0)
    assert np.all(np.diff(logtau) < 0)


def test_generator_grid_matches_the_training_grid():
    """The tau_500 grid the atmospheres are generated on equals the training grid to 1e-6 (45 nodes).

    ``load_source_arrays`` refuses an atmosphere file whose grid differs by more than 1e-6.
    """
    from scripts.base_training import TrainingConfig
    from scripts.synthesis.tau500_multi_step_regen import NEW_LOGTAU_GRID
    grid = TrainingConfig(checkpoint_dir="/tmp/_ck", log_dir="/tmp/_lg").get_logtau_values()
    assert NEW_LOGTAU_GRID.size == grid.size == 45
    np.testing.assert_allclose(np.round(NEW_LOGTAU_GRID.astype(np.float32), 6), grid, atol=1e-6)


# --- Fine-tuning -------------------------------------------------------------------------

def test_finetuning_refuses_a_base_run_without_a_balancing_node(tmp_path):
    """Fine-tuning balances |Bz| at the base run's node; without one it raises instead of using the middle node.

    The base run's settings are replayed from its experiment_config.json, which must sit
    next to the checkpoint.
    """
    import scripts.finetune as finetune
    from models.pinn_mscnn_model import PhysicsInformedMSCNN
    run = tmp_path / "exp" / "wfa_only"
    run.mkdir(parents=True)
    torch.save({"model_state_dict": PhysicsInformedMSCNN(scales=[1], output_features=135).state_dict()},
               run / "final_model.pth")
    degradation = {"spatial_psf": False, "hinode_noise": 0.0}
    (run / "experiment_config.json").write_text(json.dumps(
        {"physics_config": {"lambda_wfa": 10.0}, "data_config": {"bz_balance_tau_idx": None, **degradation}}))
    with pytest.raises(ValueError, match="bz_balance_tau_idx"):
        finetune.load_and_adapt_config(run / "final_model.pth")
    (run / "experiment_config.json").write_text(json.dumps(
        {"physics_config": {"lambda_wfa": 10.0}, "data_config": {"bz_balance_tau_idx": 44, **degradation}}))
    cfg, _ = finetune.load_and_adapt_config(run / "final_model.pth", finetune_epochs_override=3)
    assert (cfg.lambda_wfa, cfg.apply_bz_bin_balance, cfg.bz_balance_scope, cfg.n_epochs) == (10.0, True, "global", 3)
    # The base run's degradation is kept, not replaced by today's defaults.
    assert (cfg.spatial_psf, cfg.hinode_noise) == (False, 0.0)


def test_finetuning_refuses_a_base_run_without_its_degradation_settings(tmp_path):
    """A base run without ``spatial_psf`` and ``hinode_noise`` predates Phase 1b and is refused.

    Fine-tuning it with today's defaults would switch it to blurred, noisy profiles halfway
    through its training.
    """
    import scripts.finetune as finetune
    from models.pinn_mscnn_model import PhysicsInformedMSCNN
    run = tmp_path / "exp" / "wfa_only"
    run.mkdir(parents=True)
    torch.save({"model_state_dict": PhysicsInformedMSCNN(scales=[1], output_features=135).state_dict()},
               run / "final_model.pth")
    (run / "experiment_config.json").write_text(json.dumps(
        {"physics_config": {"lambda_wfa": 10.0}, "data_config": {"bz_balance_tau_idx": 44}}))
    with pytest.raises(ValueError, match="has no spatial_psf, hinode_noise"):
        finetune.load_and_adapt_config(run / "final_model.pth")


# --- Generation safeguards ---------------------------------------------------------------

def test_merge_refuses_chunks_with_unphysical_continua(tmp_path, monkeypatch):
    """The merge step refuses to write a step if any pixel's continuum is outside 0.3 to 2 (HSRA units).

    NICOLE can return garbage without any error (negative continua appeared when the local
    field slots were filled), and the merge used to reject only NaN. Two chunks tile a
    4-row frame; one pixel has a continuum of -12, so the merge must exit without writing.
    """
    import dataclasses
    import scripts.synthesis.merge_tau500_stokes as merge
    from scripts.base_training import TrainingConfig
    nx, ny, nwl, nz = 4, 3, 5, 2
    monkeypatch.setattr(merge, "TrainingConfig",
                        lambda: dataclasses.replace(TrainingConfig(checkpoint_dir=str(tmp_path / "c"),
                                                                   log_dir=str(tmp_path / "l")), nx=nx, ny=ny))
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    for r0, r1 in ((0, 2), (2, 4)):
        st = np.ones((r1 - r0, ny, nwl, 4))
        if r0 == 2:
            st[0, 1, :, 0] = -12.0
        np.save(chunks / f"chunk_150_{r0:04d}_{r1:04d}_stokes.npy", st)
        arr = np.zeros((r1 - r0, ny, nz))
        np.savez(chunks / f"chunk_150_{r0:04d}_{r1:04d}_atmos.npz", logtau=np.array([-1.0, 0.0]),
                 T=arr + 5000, Vz=arr, v_los=arr, Bz=arr, Bx=arr, By=arr, z_km=arr)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["merge", "--step", "150", "--chunk-dir", str(chunks), "--output-dir", str(out)])
    with pytest.raises(SystemExit, match="continuum outside"):
        merge.main()
    assert not (out / "stokes_150_nicole_tau500.npy").exists()
