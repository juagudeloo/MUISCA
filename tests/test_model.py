"""Tests for models/mscnn_model.py and models/pinn_mscnn_model.py.

They check the architecture's shapes, the torch denormalization used by the physics terms,
the pixel indexing of the approximation maps, and that the physics terms send gradients to
the network (a past bug made them contribute zero gradient).
"""
import contextlib
import io

import numpy as np
import pytest
import torch

from models.mscnn_model import CoarseGrain, MSCNNInversionModel
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.normalizer import MhdNormalizer

LOGTAU = np.array([-2.0, -1.0, 0.0])
N_TAU = LOGTAU.size


@pytest.fixture()
def fitted_normalizer(tmp_path):
    """MhdNormalizer fitted on a small synthetic atmosphere, returned with that atmosphere."""
    rng = np.random.default_rng(0)
    atm = {"T": 5000 + 300 * rng.standard_normal((8, 8, N_TAU)),
           "Vz": 0.5 + rng.standard_normal((8, 8, N_TAU)),
           "Bz": 200 * rng.standard_normal((8, 8, N_TAU))}
    norm = MhdNormalizer(n_tau=N_TAU, raw_values_dir=str(tmp_path))
    norm.update(atm)
    with contextlib.redirect_stdout(io.StringIO()):
        norm.finalize()
    return norm, atm


def _model(**kwargs) -> PhysicsInformedMSCNN:
    """Small physics-informed model on the 3-node grid (single-height terms)."""
    defaults = dict(scales=[1, 2], n_tau_levels=N_TAU, lambda_wfa=0.0, lambda_doppler=0.0, lambda_temp=0.0,
                    blos_physics_mode="single_height", blos_target_logtau=-1.0,
                    vlos_physics_mode="single_height", vlos_target_logtau=-1.0,
                    temp_physics_mode="single_height", temp_target_logtau=0.0)
    defaults.update(kwargs)
    return PhysicsInformedMSCNN(**defaults)


# --- Architecture ------------------------------------------------------------------------

@pytest.mark.parametrize("scales,length", [([1, 2, 3], 112), ([1], 112), ([1, 2, 3, 4], 100)])
def test_output_shape_and_flatten_size_match_the_forward_pass(scales, length):
    """The constructor's flatten size matches the real feature length, so any configuration runs.

    The paper configuration (scales 1, 2, 3; 112 points; 45 nodes) must give 135 outputs.
    """
    model = MSCNNInversionModel(scales=scales, input_length=length, output_features=135)
    assert model(torch.zeros(3, 2, length)).shape == (3, 135)


def test_coarse_grain_averages_blocks_and_drops_the_remainder():
    """Scale 3 averages samples (0, 1, 2) and (3, 4, 5); the leftover sample 6 is dropped."""
    out = CoarseGrain(3)(torch.arange(7.0).reshape(1, 1, 7))
    assert out.tolist() == [[[1.0, 4.0]]]


def test_scales_see_the_same_convolution_weights():
    """All scales share one convolution block, so the model has a single Conv1d."""
    model = MSCNNInversionModel(scales=[1, 2, 3])
    assert sum(isinstance(m, torch.nn.Conv1d) for m in model.modules()) == 1


# --- Physics context and denormalization --------------------------------------------------

def test_set_physics_context_picks_the_nearest_node(fitted_normalizer):
    """Single-height targets snap to the nearest node of the grid (here -0.8 to -1.0, 0.3 to 0.0)."""
    norm, _ = fitted_normalizer
    m = _model(blos_target_logtau=-0.8, temp_target_logtau=0.3)
    m.set_physics_context(norm, LOGTAU)
    assert m._blos_target_logtau_idx == 1 and float(m.blos_target_logtau) == -1.0
    assert m._temp_target_logtau_idx == 2 and float(m.temp_target_logtau) == 0.0


@pytest.mark.parametrize("param", ["T", "Bz"])
def test_torch_denormalization_inverts_the_normalizer(fitted_normalizer, param):
    """The torch denormalization returns the physical values the normalizer started from.

    The physics terms compare these values (K, G) with the approximations, so they must
    equal the physical targets when the prediction equals the normalized target.
    """
    norm, atm = fitted_normalizer
    m = _model()
    m.set_physics_context(norm, LOGTAU)
    z = {k: torch.tensor(v.reshape(-1, N_TAU), dtype=torch.float32) for k, v in norm.transform(atm).items()}
    pred = torch.cat([z["T"], z["Vz"], z["Bz"]], dim=1)
    out = m._denormalize_predictions(pred)[param].numpy()
    np.testing.assert_allclose(out, atm[param].reshape(-1, N_TAU), rtol=1e-4, atol=1e-2)


@pytest.mark.xfail(strict=True, reason="Known, fixed in reframe Phase 1: the 'centered' Vz transform does not "
                   "subtract the mean but the denormalization adds it, so the Doppler term sees Vz + mean(tau).")
def test_torch_denormalization_inverts_the_normalizer_for_vz(fitted_normalizer):
    """Same as for T and Bz, for Vz (km/s): the Doppler term must see the physical velocity."""
    norm, atm = fitted_normalizer
    m = _model()
    m.set_physics_context(norm, LOGTAU)
    z = {k: torch.tensor(v.reshape(-1, N_TAU), dtype=torch.float32) for k, v in norm.transform(atm).items()}
    out = m._denormalize_predictions(torch.cat([z["T"], z["Vz"], z["Bz"]], dim=1))["Vz"].numpy()
    np.testing.assert_allclose(out, atm["Vz"].reshape(-1, N_TAU), atol=1e-3)


def test_tau_average_of_a_constant_profile_is_that_constant(fitted_normalizer):
    """The linear-tau average of a profile that is the same at every node returns that value (G)."""
    norm, _ = fitted_normalizer
    m = _model(blos_physics_mode="tau_averaged")
    m.set_physics_context(norm, LOGTAU)
    assert torch.allclose(m._compute_tau_averaged_blos(torch.full((4, N_TAU), 123.0)), torch.full((4,), 123.0))


# --- Physics losses ----------------------------------------------------------------------

def _prediction_with_blos(norm, b_at_node1: np.ndarray) -> torch.Tensor:
    """Normalized prediction whose Bz at node 1 (log tau = -1) equals ``b_at_node1`` (G)."""
    n = b_at_node1.size
    bz = np.zeros((n, 1, N_TAU))
    bz[:, 0, 1] = b_at_node1
    zb = norm.transform({"T": np.full((n, 1, N_TAU), 5000.0), "Vz": np.zeros((n, 1, N_TAU)), "Bz": bz})
    pred = np.concatenate([zb["T"].reshape(n, -1), zb["Vz"].reshape(n, -1), zb["Bz"].reshape(n, -1)], axis=1)
    return torch.tensor(pred, dtype=torch.float32)


def test_wfa_loss_compares_each_sample_with_its_own_pixel(fitted_normalizer):
    """The WFA term reads the approximation at the sample's own (x, y) pixel.

    The map holds a different field in every pixel (G). A prediction equal to the map at
    each sample's (x, y) gives a loss of about 0; giving each sample another sample's
    value gives a large loss. This guards the (x, y) convention shared with
    MuramStepDataset: with the axes swapped the term would pull toward the wrong pixel.
    """
    norm, _ = fitted_normalizer
    m = _model(lambda_wfa=1.0)
    approx = np.arange(1, 33, dtype=float).reshape(4, 8) * 25.0       # (nx=4, ny=8)
    m.set_physics_context(norm, LOGTAU, blos_approx=approx)
    xy = torch.tensor([[0, 7], [3, 1], [2, 5]])
    pred = _prediction_with_blos(norm, approx[xy[:, 0], xy[:, 1]])
    assert float(m._compute_wfa_loss(m._denormalize_predictions(pred), xy)) < 1e-3
    pred_wrong = _prediction_with_blos(norm, approx[xy[:, 0], xy[:, 1]][::-1].copy())
    assert float(m._compute_wfa_loss(m._denormalize_predictions(pred_wrong), xy)) > 0.3


def test_doppler_loss_skips_failed_fits(fitted_normalizer):
    """Pixels whose Doppler fit failed (NaN) are left out; an all-NaN batch gives a loss of 0."""
    norm, _ = fitted_normalizer
    m = _model(lambda_doppler=1.0)
    vmap = np.array([[np.nan, 1.0], [np.nan, np.nan]])
    m.set_physics_context(norm, LOGTAU, vlos_approx=vmap)
    pred = torch.zeros(2, 3 * N_TAU)
    loss = m._compute_doppler_loss(m._denormalize_predictions(pred), torch.tensor([[0, 1], [1, 0]]))
    assert torch.isfinite(loss)
    assert float(m._compute_doppler_loss(m._denormalize_predictions(pred), torch.tensor([[0, 0], [1, 1]]))) == 0.0


def test_physics_terms_are_zero_when_disabled():
    """With every lambda at 0 (the baseline) the total loss equals the MSE exactly."""
    m = _model()
    out = m.compute_loss(torch.ones(4, 3 * N_TAU), torch.zeros(4, 3 * N_TAU), torch.zeros(4, 2, dtype=torch.long))
    assert float(out["physics"]) == 0.0 and float(out["loss"]) == float(out["mse"]) == 1.0


def test_physics_terms_send_gradients_to_the_network(fitted_normalizer):
    """Each physics term alone produces a non-zero gradient on the network's weights.

    Regression for the numpy round trip in the denormalization, which cut the autograd
    graph: the terms were logged but contributed zero gradient for hundreds of epochs.
    """
    norm, _ = fitted_normalizer
    for term in ("lambda_wfa", "lambda_doppler", "lambda_temp"):
        torch.manual_seed(0)
        m = _model(**{term: 1.0})
        m.set_physics_context(norm, LOGTAU, blos_approx=np.full((4, 4), 300.0),
                              vlos_approx=np.full((4, 4), 2.0), temp_approx=np.full((4, 4), 7000.0))
        x = torch.randn(5, 2, 112)
        total, _ = m.compute_physics_loss(m(x), torch.randint(0, 4, (5, 2)))
        total.backward()
        grad = sum(float(p.grad.abs().sum()) for p in m.parameters() if p.grad is not None)
        assert grad > 0, term


def test_physics_gate_closes_every_term(fitted_normalizer):
    """With the gate closed (``enable_physics=False``) no physics term contributes, for any lambda."""
    norm, _ = fitted_normalizer
    m = _model(lambda_wfa=1.0, lambda_doppler=1.0, lambda_temp=1.0)
    m.set_physics_context(norm, LOGTAU, blos_approx=np.ones((2, 2)), vlos_approx=np.ones((2, 2)),
                          temp_approx=np.ones((2, 2)))
    out = m.compute_loss(torch.zeros(3, 3 * N_TAU), torch.zeros(3, 3 * N_TAU), torch.zeros(3, 2, dtype=torch.long),
                         enable_physics=False)
    assert float(out["physics"]) == 0.0


def test_model_accepts_four_stokes_channels():
    """The architecture already takes I, Q, U, V: ``in_channels=4`` gives the same output shape (Phase 1c)."""
    model = PhysicsInformedMSCNN(scales=[1, 2, 3], in_channels=4, output_features=135)
    assert model(torch.zeros(3, 4, 112)).shape == (3, 135)
    assert next(model.msfm.conv_block.c1.parameters()).shape[1] == 4
