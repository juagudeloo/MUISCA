"""Targets for the HMI/MDI-like velocity approximation and its loss term (reframe plan, Phase 1e).

The approximation follows the MDI/HMI algorithm (Couvidat et al.): the first Fourier
coefficients of I + V and I - V over a window around Fe I 6301.5 A give a phase, the
phase gives a velocity for each circular polarization, and the line-of-sight velocity is
their mean, corrected with a lookup table built by shifting the mean MURaM profile by
known velocities. Positive is a redshift, the convention of the training target, SPINOR
and the Doppler approximation.

Every test is a strict xfail until Phase 1e implements ``compute_vlos_mdi``,
``VLOS_MDI_METHOD`` and the ``mdi`` loss term.
"""
import contextlib
import io

import astropy.units as u
import numpy as np
import pytest
import torch

from utils.hinode_wavelengths import hinode_wavelength_grid
from utils.physics_utils import ApproxInversions

TARGET = "Agreed in the reframe plan (Phase 1e): MDI-like velocity flavor, not implemented yet."
WL = hinode_wavelength_grid()
LAMBDA0 = 6301.5012
C_KMS = 299792.458


def _line(v_kms: float, split_aa: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """I and V of a Gaussian Fe I 6301.5 A line shifted by ``v_kms``; ``split_aa`` splits the two circular components."""
    center = LAMBDA0 * (1 + v_kms / C_KMS)
    sig, depth = 0.05, 0.6
    rcp = 1 - depth * np.exp(-((WL - center - split_aa) ** 2) / (2 * sig ** 2))
    lcp = 1 - depth * np.exp(-((WL - center + split_aa) ** 2) / (2 * sig ** 2))
    return 0.5 * (rcp + lcp), 0.5 * (rcp - lcp)


def _mdi(I: np.ndarray, V: np.ndarray) -> float:
    """Run compute_vlos_mdi on one pixel and return km/s."""
    inv = ApproxInversions({"I": I[None, None], "V": V[None, None]}, WL, LAMBDA0 * u.Angstrom, 1.67)
    return float(inv.compute_vlos_mdi().to_value(u.km / u.s)[0, 0])


@pytest.mark.xfail(strict=True, reason=TARGET)
@pytest.mark.parametrize("v_kms", [-1.5, 0.0, 1.0])
def test_mdi_recovers_a_known_shift_with_the_redshift_sign(v_kms):
    """A line shifted by v (km/s) gives v back within 50 m/s, positive for a redshift."""
    I, V = _line(v_kms)
    assert _mdi(I, V) == pytest.approx(v_kms, abs=0.05)


@pytest.mark.xfail(strict=True, reason=TARGET)
def test_mdi_is_insensitive_to_zeeman_splitting():
    """Splitting the line into its two circular components (a longitudinal field) does not change v.

    The two polarizations are shifted in opposite directions by the field, so averaging
    the velocities from I + V and I - V cancels the Zeeman shift, which is the point of
    the MDI scheme. A 30 mA splitting (about 1.4 km/s per component) must change v by
    less than 50 m/s.
    """
    I0, V0 = _line(0.8)
    I1, V1 = _line(0.8, split_aa=0.03)
    assert _mdi(I1, V1) == pytest.approx(_mdi(I0, V0), abs=0.05)


@pytest.mark.xfail(strict=True, reason=TARGET)
def test_mdi_method_tag_is_part_of_both_cache_signatures():
    """A ``VLOS_MDI_METHOD`` tag exists and enters both cache signatures, so a change of algorithm rebuilds the caches."""
    from scripts.base_training import TrainingConfig, build_balanced_cache_signature, build_cache_config_signature
    from utils.physics_utils import VLOS_MDI_METHOD
    cfg = TrainingConfig()
    assert VLOS_MDI_METHOD in build_cache_config_signature(cfg).values()
    assert VLOS_MDI_METHOD in build_balanced_cache_signature(cfg, [110]).values()


@pytest.mark.xfail(strict=True, reason=TARGET)
def test_mdi_loss_term_is_weighted_logged_and_sends_gradients(tmp_path):
    """With ``lambda_mdi`` > 0 the loss gains an ``mdi`` term that is logged and moves the network's weights.

    The term is an RRMSE between the predicted v_LOS at ``mdi_target_logtau`` and the MDI
    map of the sample's pixel, like the Doppler term.
    """
    from models.pinn_mscnn_model import PhysicsInformedMSCNN
    from utils.normalizer import MhdNormalizer
    rng = np.random.default_rng(0)
    atm = {"T": 5000 + 100 * rng.standard_normal((4, 4, 3)), "Vz": rng.standard_normal((4, 4, 3)),
           "Bz": 100 * rng.standard_normal((4, 4, 3))}
    norm = MhdNormalizer(n_tau=3, raw_values_dir=str(tmp_path))
    norm.update(atm)
    with contextlib.redirect_stdout(io.StringIO()):
        norm.finalize()
    m = PhysicsInformedMSCNN(scales=[1], n_tau_levels=3, lambda_wfa=0.0, lambda_doppler=0.0, lambda_temp=0.0,
                             lambda_mdi=1.0, mdi_target_logtau=-1.0)
    m.set_physics_context(norm, np.array([-2.0, -1.0, 0.0]), mdi_approx=np.full((4, 4), 1.5))
    total, parts = m.compute_physics_loss(m(torch.randn(5, 2, 112)), torch.randint(0, 4, (5, 2)))
    assert "mdi" in parts and parts["mdi"] > 0
    total.backward()
    assert sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None) > 0
