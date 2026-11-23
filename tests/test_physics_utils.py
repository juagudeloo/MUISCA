"""Tests for the WFA and Doppler approximations in utils/physics_utils.py.

The black-body approximation has its own file (tests/test_blackbody.py).
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.constants import c, e, m_e

from utils.hinode_wavelengths import hinode_wavelength_grid
from utils.physics_utils import ApproxInversions

WL = hinode_wavelength_grid()          # 112 Hinode/SP points, Angstrom
LAMBDA0 = 6301.5                        # Fe I line used by the WFA and Doppler terms, Angstrom
G_EFF = 1.67
WL_RANGE = [15, 60]                     # the training configuration (TrainingConfig.wl_range)
C_WFA = (e.si / (4 * np.pi * m_e * c)).to(1 / u.G / u.Angstrom).value  # 4.67e-13 A^-1 G^-1


def _line(shift_aa: float = 0.0, depth: float = 0.6, sigma_aa: float = 0.05) -> np.ndarray:
    """Gaussian absorption line at LAMBDA0 + shift on the Hinode grid, continuum 1."""
    return 1.0 - depth * np.exp(-((WL - LAMBDA0 - shift_aa) ** 2) / (2 * sigma_aa ** 2))


@pytest.mark.parametrize("b_gauss", [-800.0, 50.0, 500.0])
def test_wfa_recovers_field_and_sign(b_gauss):
    """The WFA fit returns the field (G) that generated V, with its sign, to 1e-6 relative.

    V is built exactly as the weak-field approximation assumes,
    V = -C lambda0^2 g_eff B dI/dlambda with C = e / (4 pi m_e c) = 4.67e-13 A^-1 G^-1,
    using the same finite-difference derivative as the code. A negative, a weak and a
    strong field check that the sign is preserved and that nothing clips or thresholds B.
    The physics loss compares the network's B_LOS with this value, so a sign or a factor
    error here would push the network toward a wrong field.
    """
    I = _line()
    dI = np.gradient(I) / np.gradient(WL)
    V = -C_WFA * LAMBDA0 ** 2 * G_EFF * b_gauss * dI
    inv = ApproxInversions({"I": I[None, None], "V": V[None, None]}, WL, LAMBDA0 * u.Angstrom, G_EFF)
    b = inv.compute_blos_wfa(WL_RANGE)
    assert b.unit == u.G
    assert b.value[0, 0] == pytest.approx(b_gauss, rel=1e-6)


def test_wfa_ignores_constant_offset():
    """A constant offset in V (2e-3 continuum units) does not change the fitted field.

    The least-squares fit includes an offset term, so a flat V bias such as residual
    crosstalk must not leak into B. The field must still come back as 300 G.
    """
    I = _line()
    dI = np.gradient(I) / np.gradient(WL)
    V = -C_WFA * LAMBDA0 ** 2 * G_EFF * 300.0 * dI + 2e-3
    inv = ApproxInversions({"I": I[None, None], "V": V[None, None]}, WL, LAMBDA0 * u.Angstrom, G_EFF)
    assert inv.compute_blos_wfa(WL_RANGE).value[0, 0] == pytest.approx(300.0, rel=1e-6)


@pytest.mark.parametrize("shift_aa", [-0.03, 0.0, 0.02])
def test_doppler_sign_and_value(shift_aa):
    """A line shifted by d_lambda gives v = c d_lambda / lambda0 (km/s) within 1 m/s.

    Positive v must mean a redshift, the MURaM Vz convention the network is trained on.
    The shifts -30, 0 and +20 mA correspond to about -1.43, 0 and +0.95 km/s. A sign error
    here would make the Doppler loss pull the network toward the opposite velocity.
    """
    I = _line(shift_aa)
    inv = ApproxInversions({"I": I[None, None], "V": np.zeros((1, 1, WL.size))}, WL,
                           LAMBDA0 * u.Angstrom, G_EFF)
    v = inv.compute_vlos_doppler(WL_RANGE)
    assert v.unit == u.km / u.s
    expected = shift_aa / LAMBDA0 * c.to_value(u.km / u.s)
    assert v.value[0, 0] == pytest.approx(expected, abs=1e-3)


def test_compare_with_mhd_metrics(tmp_path):
    """Per-node RRMSE and Pearson correlation match a direct numpy computation.

    RRMSE = sqrt(mean((approx - truth)^2)) / mean(|truth|) is dimensionless. These metrics
    choose the comparison height of each physics loss, so they must be exact. The synthetic
    approximation is node 1 plus noise, so node 1 (log tau = -1) must win both metrics.
    """
    rng = np.random.default_rng(0)
    nx, ny, nz = 6, 5, 3
    truth = rng.standard_normal((nx, ny, nz)) * 100.0
    inv = ApproxInversions({"I": np.ones((nx, ny, WL.size)), "V": np.zeros((nx, ny, WL.size))},
                           WL, LAMBDA0 * u.Angstrom, G_EFF)
    inv.blos = (truth[:, :, 1] + rng.standard_normal((nx, ny)) * 10.0) * u.G
    out = inv.compare_with_mhd({"Bz": truth}, approximation="blos",
                               logtau_values=np.array([-2.0, -1.0, 0.0]), save_dir=str(tmp_path))
    for k in range(nz):
        rrmse = np.sqrt(np.mean((inv.blos.value - truth[:, :, k]) ** 2)) / np.mean(np.abs(truth[:, :, k]))
        corr = np.corrcoef(inv.blos.value.ravel(), truth[:, :, k].ravel())[0, 1]
        assert out["rrmse_values"][k] == pytest.approx(rrmse)
        assert out["correlation_values"][k] == pytest.approx(corr)
    assert out["min_rrmse_logtau"] == -1.0 and out["max_corr_logtau"] == -1.0
