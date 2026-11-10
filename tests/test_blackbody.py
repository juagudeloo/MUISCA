"""Tests for the black-body temperature approximation (exact Planck inversion, HSRA anchor).

Run with `pytest -s tests/test_blackbody.py` to see the sanity-check table.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import astropy.units as u
from astropy.constants import h, c, k_B

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.physics_utils import (  # noqa: E402
    ApproxInversions,
    HSRA_REFERENCE_WAVELENGTH,
    hsra_continuum_intensity,
    hsra_brightness_temperature,
    planck_brightness_temperature,
)
from utils.hinode_wavelengths import hinode_wavelength_grid  # noqa: E402

CONT_INDICES = [108, 109, 110, 111]
WL = hinode_wavelength_grid()
LAMBDA_C = np.mean(WL[CONT_INDICES]) * u.Angstrom


def _bb_temperature(i_norm: np.ndarray) -> np.ndarray:
    """Run compute_temperature_blackbody on a flat continuum of the given HSRA-normalized levels."""
    i_norm = np.atleast_1d(np.asarray(i_norm, dtype=float))
    cube = np.repeat(i_norm[:, None, None], WL.size, axis=2)  # (n, 1, n_wl)
    inv = ApproxInversions(
        stokes={"I": cube, "V": np.zeros_like(cube)},
        wavelength=WL,
        central_wavelength=6301.5 * u.Angstrom,
        lande_factor=1.67,
    )
    return inv.compute_temperature_blackbody(cont_indices=CONT_INDICES).to_value(u.K)[:, 0]


def _t_ref() -> float:
    """Brightness temperature at lambda_c of the HSRA reference intensity (I_norm = 1)."""
    return planck_brightness_temperature(
        hsra_continuum_intensity(HSRA_REFERENCE_WAVELENGTH), LAMBDA_C
    ).to_value(u.K)


def test_unit_continuum_returns_reference_temperature():
    assert _bb_temperature(1.0)[0] == pytest.approx(_t_ref(), abs=1e-6)
    # Sanity on the anchor itself: HSRA at 6302 A is ~6125 K
    assert hsra_brightness_temperature().to_value(u.K) == pytest.approx(6125.0, abs=5.0)


def test_small_contrasts_match_first_order_expansion():
    t_ref = _t_ref()
    x = (h * c / (LAMBDA_C * k_B * t_ref * u.K)).decompose().value
    alpha = x * np.exp(x) / np.expm1(x)
    i_norm = np.linspace(0.95, 1.05, 21)
    linear = t_ref * (1.0 + (i_norm - 1.0) / alpha)
    assert np.max(np.abs(_bb_temperature(i_norm) - linear)) < 5.0


def test_monotonic_in_intensity():
    t = _bb_temperature(np.linspace(0.1, 2.0, 400))
    assert np.all(np.diff(t) > 0)


def test_rejects_non_hsra_normalized_profiles():
    inv = ApproxInversions(
        stokes={"I": np.ones((1, 1, WL.size))},
        wavelength=WL,
        central_wavelength=6301.5 * u.Angstrom,
        lande_factor=1.67,
    )
    with pytest.raises(ValueError):
        inv.compute_temperature_blackbody(hsra_normalized=False)


def test_sanity_against_muram_step_110():
    """Median/sd of (T_BB - T_sim) on a MURaM training step, through the real Stokes pipeline."""
    from scripts.base_training import TrainingConfig, load_source_arrays

    config = TrainingConfig()
    if config.data_source != "nicole_tau500":
        pytest.skip("sanity check is defined for the nicole_tau500 data source")
    result = load_source_arrays(step=110, config=config, ignore_missing_files=True)
    if result is None:
        pytest.skip("step 110 nicole_tau500 files not available")
    stokes, mhd = result

    # 1-in-4 spatial subsample keeps the LSF + cubic resampling fast
    sub = (slice(None, None, 4), slice(None, None, 4))
    stokes.data = {k: np.ascontiguousarray(v[sub]) for k, v in stokes.data.items() if k in ("I", "V")}
    stokes.nx, stokes.ny = stokes.data["I"].shape[:2]
    t_sim = mhd["T"].to_value(u.K)[sub]

    stokes.load_hinode_lsf(config.data_path / config.lsf_path)
    stokes.apply_spectral_convolution()
    stokes.resample_to_hinode()

    inv = ApproxInversions(
        stokes=stokes.data,
        wavelength=stokes.wl,
        central_wavelength=config.central_wavelength * u.Angstrom,
        lande_factor=config.lande_factor,
    )
    t_bb = inv.compute_temperature_blackbody(cont_indices=config.temp_continuum_indices).to_value(u.K)

    logtau = np.round(config.get_logtau_values(), 2)
    old = {0.0: -382, -0.1: -185, -0.2: -9}  # previous method (scene mean -> 6000 K, linearized)
    print("\ncontinuum indices", config.temp_continuum_indices, "->",
          np.round(stokes.wl[config.temp_continuum_indices], 4), "A")
    print("old default [0, 1, 2, 3] ->", np.round(stokes.wl[[0, 1, 2, 3]], 4), "A")
    print(f"T_ref (I_norm = 1) = {_t_ref():.1f} K")
    print(f"{'log tau':>8} | {'median T_sim':>12} | {'old median':>10} | {'new median':>10} | {'new sd':>6} | {'corr':>5}")
    medians = {}
    for lt in (0.0, -0.1, -0.2):
        k = int(np.where(np.isclose(logtau, lt))[0][0])
        diff = t_bb - t_sim[..., k]
        medians[lt] = float(np.median(diff))
        corr = np.corrcoef(t_bb.ravel(), t_sim[..., k].ravel())[0, 1]
        print(f"{lt:>8.1f} | {np.median(t_sim[..., k]):>12.0f} | {old[lt]:>+10d} | "
              f"{medians[lt]:>+10.0f} | {np.std(diff):>6.0f} | {corr:>5.3f}")

    assert abs(medians[-0.2]) < 100.0
