"""Tests for utils/tau500_opacity.py (H- continuum opacity at 5000 A used for the tau_500 scale)."""
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from utils.tau500_opacity import (
    K_BOLTZMANN,
    _h_ionization_fractions,
    electron_pressure_hsra,
    kappa_500,
)

HSRA_DIR = Path(__file__).resolve().parent.parent / "data" / "csv"
needs_hsra = pytest.mark.skipif(not (HSRA_DIR / "hsra_pgas.model").exists(), reason="HSRA model files absent")

M_H = 1.6735e-24      # g
MU = 1.3              # mean molecular weight of neutral solar gas
G_SUN = 2.74e4        # cm s^-2


def _rho(T, P):
    """Ideal-gas density rho = P mu m_H / (k T) in g cm^-3, for P in dyn cm^-2 and T in K."""
    return P * MU * M_H / (K_BOLTZMANN * T)


def test_kappa_positive_and_increasing_with_temperature():
    """kappa_500 (cm^2 g^-1) is positive and grows strictly with T at fixed pressure.

    Hotter gas ionizes more metals, so it has more free electrons and more H-. The test
    scans 4500 to 7000 K at P = 1e5 dyn cm^-2 (photospheric). A non-monotonic opacity
    would make the tau_500 scale of hot and cold MURaM columns inconsistent.
    """
    T = np.linspace(4500.0, 7000.0, 26)
    P = np.full_like(T, 1.0e5)  # dyn cm^-2, photospheric
    k = kappa_500(T, P, _rho(T, P))
    assert np.all(k > 0)
    assert np.all(np.diff(k) > 0)


def test_saha_fractions_are_physical():
    """The Saha hydrogen fractions are physical in the photosphere.

    At 4500, 5800 and 6500 K, with the HSRA electron pressure, neutral hydrogen must be
    between 99 % and 100 % of all hydrogen and H- must be a trace (positive, below 1e-6).
    A wrong unit in the Saha constant would move these fractions by orders of magnitude.
    """
    T = np.array([4500.0, 5800.0, 6500.0])
    p1, p10 = _h_ionization_fractions(T, electron_pressure_hsra(T))
    assert np.all((p1 > 0.99) & (p1 <= 1.0))         # hydrogen is almost all neutral
    assert np.all((p10 > 0) & (p10 < 1e-6))         # H- is a trace species (~1e-8)


@needs_hsra
@pytest.mark.data
def test_electron_pressure_fit_reproduces_hsra():
    """The fitted Pe(T) (dyn cm^-2) matches the HSRA table within 0.1 % for log tau >= -4.

    The H- opacity is roughly proportional to Pe, so an error in this fit goes straight
    into the tau_500 scale. Needs the HSRA files in data/csv.
    """
    b = np.loadtxt(HSRA_DIR / "hsra_pel.model", skiprows=2)
    deep = b[:, 0] >= -4.0
    np.testing.assert_allclose(electron_pressure_hsra(b[deep, 1]), b[deep, 2], rtol=1e-3)


@needs_hsra
@pytest.mark.data
def test_tau500_scale_reproduces_hsra():
    """Integrating our kappa_500 over the HSRA gives back the HSRA's own log tau.

    Uses hydrostatic equilibrium, dtau = kappa dP / g, anchored at log tau = -4. The
    result must agree within 0.04 dex for -3 <= log tau <= 0.5. Deeper it falls short
    (-0.09 dex at +1.0, -0.17 at +1.4) because only H- is included; the test checks that
    the deepest point stays between -0.25 and 0 dex. This validates the tau_500 remap of
    the MURaM cubes. Needs the HSRA files in data/csv.
    """
    a = np.loadtxt(HSRA_DIR / "hsra_pgas.model", skiprows=2)
    a = a[a[:, 0] >= -4.0]
    lt, T, P = a[:, 0], a[:, 1], a[:, 2]
    tau = 10 ** lt[0] + cumulative_trapezoid(kappa_500(T, P, _rho(T, P)) / G_SUN, P, initial=0.0)
    diff = np.log10(tau) - lt
    photosphere = (lt >= -3.0) & (lt <= 0.5)
    assert np.max(np.abs(diff[photosphere])) < 0.04
    assert -0.25 < diff[np.argmin(np.abs(lt - 1.4))] < 0.0
