"""Approximate continuum opacity at 500 nm (tau_500), for the --tau500 experiment.

This module now defines the optical depth scale of all training data: the
``nicole_tau500`` cubes are remapped onto log(tau_500) with ``kappa_500`` (see
``scripts/synthesis/tau500_multi_step_regen.compute_logtau500``). Validation:
integrating ``kappa_500`` hydrostatically over the HSRA, anchored at log tau = -4,
reproduces the HSRA's own log tau within 0.04 dex for -3 <= log tau <= 0.5, and falls
0.09 / 0.17 dex short at log tau = 1.0 / 1.4 (``tests/test_tau500_opacity.py``).

Examples:
    >>> import numpy as np
    >>> T, P = np.array([5000.0, 6500.0]), np.array([1.0e5, 1.0e5])   # K, dyn cm^-2
    >>> rho = P * 1.3 * 1.6735e-24 / (K_BOLTZMANN * T)                 # g cm^-3
    >>> k = kappa_500(T, P, rho)                                      # cm^2 g^-1
    >>> bool(k[1] > k[0] > 0)
    True

Original design notes follow.

This is a targeted, deliberately-scoped experiment (see docs discussion),
NOT a general-purpose stellar-atmosphere opacity code. It exists to test a
specific hypothesis: that utils/muram_data.py's existing optical-depth
pipeline uses ROSSELAND-MEAN opacity (data/csv/kappa.0.dat, see
MhdData.load_opacity_table's own "Rosseland mean opacity" comment and
notebooks/1-muram_mhd_data.ipynb's markdown), while NICOLE's .model format
expects log(tau_5000) -- a different physical depth scale -- and that this
mismatch explains an observed line-depth discrepancy between NICOLE-
synthesized and true MURaM Stokes profiles.

Physics and approximations, in order of how much they matter:

1. H-minus bound-free + free-free opacity ONLY (no He-, H2-, metals, or
   scattering). H- dominates continuum opacity at 500 nm through most of
   the photosphere; the terms skipped here matter most in the coolest
   uppermost layers. Formula ported line-for-line from NICOLE's own
   Wittmann_opac (NICOLE_v16.06/forward/wittmann_opac.f90, lines 209-251)
   at fixed lambda=5000 Angstrom, rather than from a remembered textbook
   formula -- this is the exact opacity NICOLE itself would compute for
   this term.
2. Electron pressure Pe(T) comes from an empirical fit to the real HSRA
   reference atmosphere (NICOLE_v16.06/run/hsra.model + test/conv2/
   hsra_pg.model, copied into data/csv/hsra_pel.model / hsra_pgas.model),
   using only HSRA's monotonic deep-photosphere branch (log(tau_500) in
   [-4.0, +1.4], T in [4170, 9560] K -- HSRA's temperature-minimum and
   chromospheric branch is excluded since it's non-monotonic in T and MURaM's
   box doesn't reach it). This sidesteps hand-rolling a multi-species
   (H + metals) ionization-equilibrium solve for Pe, which is what actually
   sets Pe in the cool photosphere (not hydrogen alone).
3. H ionization (H -> H+ + e-) and H- formation (H- -> H + e-) fractions
   are each a single standard Saha equation, using the Pe(T) from (2) as
   the electron pressure. Standard atomic constants (chi_H = 13.5984 eV,
   chi_H- = 0.754 eV, g_HI=2, g_HII=1, g_H-=1) -- not from NICOLE's source,
   but this is textbook physics (e.g. Gray 2005, Mihalas 1978), low risk.
4. Total hydrogen-species pressure P_Htot = Pgas - Pe (i.e. everything
   that isn't electrons is treated as hydrogen -- ignores He, ~10% of
   particles by number, and trace metals). Only used for the final
   kappa = (opacity per Htot particle) * n_Htot / rho unit conversion.

Interface mirrors utils/muram_data.py's Rosseland kappa_interp: given T (K)
and Pgas (dyn/cm^2), returns mass opacity kappa_500 in cm^2/g.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# ---- Physical constants (cgs) ----
K_BOLTZMANN = 1.380649e-16  # erg/K
M_ELECTRON = 9.10938e-28  # g
H_PLANCK = 6.62607e-27  # erg s
EV_TO_ERG = 1.602176634e-12

CHI_H_EV = 13.5984  # H ionization potential
CHI_HMINUS_EV = 0.754  # H- electron affinity


def _saha_phi(T: np.ndarray) -> np.ndarray:
    """Temperature factor of the Saha equation, ``(2 pi m_e k T / h^2)^1.5``, in cm^-3.

    Args:
        T: Temperature in K.

    Returns:
        Array with the shape of ``T``, in cm^-3.

    Examples:
        >>> f"{_saha_phi(5000.0):.3e}"
        '8.537e+20'
    """
    return (2.0 * np.pi * M_ELECTRON * K_BOLTZMANN * T / H_PLANCK**2) ** 1.5


def _load_hsra_pe_fit():
    """Build a log10(Pe) vs T interpolator from the HSRA's monotonic deep branch.

    Reads ``data/csv/hsra_pel.model``, the HSRA in NICOLE's ASCII .model format, where
    column 1 is log(tau_500), column 2 is T (K) and column 3 is the electron pressure
    (dyn cm^-2). Only points with log tau >= -4 are used, where T rises monotonically
    with depth, so Pe(T) is single-valued. ``hsra_pgas.model`` has the same layout with
    the gas pressure and is not used here (MURaM's own Pgas is used downstream).

    Returns:
        ``scipy.interpolate.interp1d`` mapping T (K) to log10(Pe / dyn cm^-2), linear,
        extrapolating outside the HSRA range of 4170-9560 K.

    Examples:
        >>> fit = _load_hsra_pe_fit()
        >>> round(float(10 ** fit(6000.0)), 1)   # dyn cm^-2  # doctest: +SKIP
        24.5
    """
    csv_dir = Path(__file__).resolve().parent.parent / "data" / "csv"
    pel_path = csv_dir / "hsra_pel.model"

    logtau, T, Pel = [], [], []
    with open(pel_path) as f:
        lines = f.readlines()[2:]  # skip "Format version" + macroturb/stray header lines
    for line in lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        logtau.append(float(parts[0]))
        T.append(float(parts[1]))
        Pel.append(float(parts[2]))
    logtau = np.asarray(logtau)
    T = np.asarray(T)
    Pel = np.asarray(Pel)

    # Deep/monotonic branch only: log(tau_500) >= -4.0 (T rises monotonically
    # with depth from the temperature minimum onward -- see module docstring).
    mask = logtau >= -4.0
    T_branch = T[mask]
    logPe_branch = np.log10(Pel[mask])

    order = np.argsort(T_branch)
    return interp1d(
        T_branch[order], logPe_branch[order],
        kind="linear", bounds_error=False, fill_value="extrapolate",
    )


_PE_FIT = _load_hsra_pe_fit()


def electron_pressure_hsra(T: np.ndarray) -> np.ndarray:
    """Electron pressure as a function of temperature, from the HSRA.

    It replaces a multi-species ionization balance (H plus metals, which actually set
    Pe in the photosphere) with the empirical Pe(T) relation of the HSRA. It reproduces
    the HSRA's own Pe to < 0.1 % on its deep branch.

    Args:
        T: Temperature in K (array-like).

    Returns:
        Pe in dyn cm^-2, float64 array.

    Examples:
        >>> import numpy as np
        >>> pe = electron_pressure_hsra(np.array([5000.0, 6500.0]))
        >>> bool(pe[1] > pe[0] > 0)
        True
    """
    T = np.asarray(T, dtype=np.float64)
    return 10.0 ** _PE_FIT(T)


def _h_ionization_fractions(T: np.ndarray, Pe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Neutral-H and H- fractions of all hydrogen, from two Saha equations.

    Uses::

        n_H+ n_e / n_HI  = phi(T) exp(-chi_H / kT)          (2 g_HII / g_HI = 1)
        n_HI n_e / n_H-  = 4 phi(T) exp(-chi_H- / kT)       (2 g_HI / g_H- = 4)
        n_e = Pe / (k T)

    with chi_H = 13.5984 eV and chi_H- = 0.754 eV.

    Args:
        T: Temperature in K.
        Pe: Electron pressure in dyn cm^-2.

    Returns:
        ``(n_HI / n_Htot, n_H- / n_Htot)``, both dimensionless. In the photosphere the
        first is ~1 and the second ~1e-8.

    Examples:
        >>> import numpy as np
        >>> p1, p10 = _h_ionization_fractions(np.array([5800.0]), np.array([30.0]))
        >>> bool(p1[0] > 0.99 and 0 < p10[0] < 1e-6)
        True
    """
    kT = K_BOLTZMANN * T
    n_e = Pe / kT
    phi = _saha_phi(T)

    ratio_ion = phi * np.exp(-CHI_H_EV * EV_TO_ERG / kT)  # n_Hplus/n_HI * n_e
    x = ratio_ion / n_e  # n_Hplus / n_HI

    ratio_hminus = 4.0 * phi * np.exp(-CHI_HMINUS_EV * EV_TO_ERG / kT)  # n_HI/n_Hminus * n_e
    y = n_e / ratio_hminus  # n_Hminus / n_HI

    denom = 1.0 + x + y
    p1 = 1.0 / denom  # n_HI / n_Htot
    p10 = y / denom  # n_Hminus / n_Htot
    return p1, p10


def _hminus_opacity_per_Htot(T: np.ndarray, Pe: np.ndarray, p1: np.ndarray, p10: np.ndarray,
                              lambda_angstrom: float = 5000.0) -> np.ndarray:
    """H- bound-free plus free-free opacity per hydrogen particle, at one wavelength.

    A line-by-line port of the H- block of NICOLE's ``Wittmann_opac``
    (``NICOLE_v16.06/forward/wittmann_opac.f90``, lines 209-251), including the
    stimulated-emission factor ``1 - exp(-h c / (lambda k T))``.

    Args:
        T: Temperature in K.
        Pe: Electron pressure in dyn cm^-2.
        p1: n_HI / n_Htot (from ``_h_ionization_fractions``).
        p10: n_H- / n_Htot.
        lambda_angstrom: Wavelength in A. 5000 for tau_500.

    Returns:
        Opacity per H-total particle, in the units of the Fortran ``kac`` before its
        final ``* P_Htot / (k T)`` conversion, which ``kappa_500`` applies.

    Examples:
        >>> import numpy as np
        >>> T = np.array([5800.0]); pe = np.array([30.0])
        >>> p1, p10 = _h_ionization_fractions(T, pe)
        >>> bool(_hminus_opacity_per_Htot(T, pe, p1, p10)[0] > 0)
        True
    """
    lam_cm = lambda_angstrom * 1.0e-8
    theta = 5040.0 / T

    x10003 = 1.4388 / lam_cm
    f1 = x10003 / T
    z1 = np.where(f1 < 1.0e-3, f1, 1.0 - np.exp(-f1))

    # Bound-free (only defined for lambda <= 16418.9 Angstrom; always true at 5000A)
    x10005 = 1.0e5 * lam_cm  # = lambda_angstrom * 1e-3
    if lambda_angstrom <= 14200.0:
        x = x10005
        cbfree = 1.0e-17 * (
            6.80133e-3 + x * (1.78708e-1 + x * (1.6479e-1 - x * (2.04842e-2 - 5.95244e-4 * x)))
        )
    else:
        x = 16.419 - x10005
        cbfree = 1.0e-17 * x * (2.69818e-1 + x * (2.2019e-1 - x * (4.11288e-2 - 2.73236e-3 * x)))

    # Free-free
    x10004 = 1.0e2 * lam_cm  # = lambda_angstrom * 1e-6
    b1 = 11.924 - 5.939 * theta
    c1 = 7.0355 - theta * 3.4592e-1
    c2 = x10005 * (-4.0192e-1 + theta * c1)
    b2 = x10004 * (-3.2062 + theta * b1 + c2)
    b3 = 2.7039e-2 * theta - 1.1493e-2 + b2
    b4 = 5.3666e-3 + theta * b3

    # bf term (cbfree*p10*z1/pe) and ff coefficient (1e-26*b4) are both
    # multiplied by (pe*p1) at the end -- see module docstring derivation.
    return p1 * (cbfree * p10 * z1 + 1.0e-26 * b4 * Pe)


def kappa_500(T: np.ndarray, Pgas: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """H--only mass opacity at 5000 A, used to build the tau_500 scale of the training data.

    The chain: Pe(T) from the HSRA, then pure-hydrogen Saha fractions, then the Wittmann
    H- bound-free and free-free opacity per H particle, then the volume opacity
    ``kac * n_Htot`` (cm^-1), with ``n_Htot = (Pgas - Pe) / kT``, then divided by rho.
    Pe is capped at ``0.999 * Pgas``. The optical depth then follows as
    ``tau_500(z) = int_z^top kappa_500 rho dz'``.

    Args:
        T: Temperature in K.
        Pgas: Gas pressure in dyn cm^-2.
        rho: Mass density in g cm^-3.

        All three are broadcastable arrays.

    Returns:
        Mass opacity in cm^2 g^-1, float64.

    Note:
        Only H- is included, with no He-, H2-, metals or scattering. That is accurate in
        the photosphere (see the HSRA test in the module docstring) and underestimates
        tau below log tau ~ 1. The interface mirrors the Rosseland ``kappa.0.dat`` path
        of ``MhdData``, except that rho is passed in, because this opacity is computed per
        point.

    Examples:
        >>> import numpy as np
        >>> T, P = np.array([6000.0]), np.array([1.0e5])
        >>> rho = P * 1.3 * 1.6735e-24 / (K_BOLTZMANN * T)
        >>> f"{kappa_500(T, P, rho)[0]:.3f}"   # cm^2 g^-1
        '0.381'
    """
    T = np.asarray(T, dtype=np.float64)
    Pgas = np.asarray(Pgas, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)

    Pe = electron_pressure_hsra(T)
    Pe = np.minimum(Pe, 0.999 * Pgas)  # guard against Pe >= Pgas at the coolest/thinnest points

    p1, p10 = _h_ionization_fractions(T, Pe)
    kac = _hminus_opacity_per_Htot(T, Pe, p1, p10)

    PHtot = Pgas - Pe
    n_Htot = PHtot / (K_BOLTZMANN * T)  # cm^-3
    kappa_cm_inverse = kac * n_Htot  # volume opacity, cm^-1 (NICOLE's wittmann_opac convention)

    return kappa_cm_inverse / rho
