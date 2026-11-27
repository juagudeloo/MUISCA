"""Tests for utils/normalizer.py (MhdNormalizer, StokesNormalizer).

Each test checks a mathematical property of the normalization, so a regression like the
B0-rescale bug fixed in commit 4cb32af (Bz std ~3.5x too large) fails loudly.
"""
import numpy as np
import pytest

from utils.normalizer import MhdNormalizer, StokesNormalizer

N_TAU = 3


def _atmosphere(seed: int = 0, nx: int = 8, ny: int = 6) -> dict:
    """Small synthetic (nx, ny, n_tau) atmosphere with tau-dependent scales."""
    rng = np.random.default_rng(seed)
    tau_scale = np.array([1.0, 2.0, 4.0])
    return {
        "T": 5000.0 + 300.0 * tau_scale * rng.standard_normal((nx, ny, N_TAU)),
        "Vz": 0.5 + tau_scale * rng.standard_normal((nx, ny, N_TAU)),
        "Bz": 50.0 * tau_scale * rng.standard_normal((nx, ny, N_TAU)),
    }


@pytest.fixture()
def fitted(tmp_path):
    """MhdNormalizer fitted once on the synthetic atmosphere, returned as ``(normalizer, atmosphere)``."""
    atm = _atmosphere()
    norm = MhdNormalizer(n_tau=N_TAU, raw_values_dir=str(tmp_path), bz_hist_bins=20000)
    norm.update(atm)
    norm.finalize()
    return norm, atm


def test_temperature_stats_match_numpy(fitted):
    """The streamed per-node T mean and std equal numpy's (relative 1e-9).

    The normalizer accumulates statistics chunk by chunk over whole MURaM steps. This
    checks that the accumulation is exact and that ``std`` is the population standard
    deviation (``ddof=0``), which is what ``transform`` divides by.
    """
    norm, atm = fitted
    for k in range(N_TAU):
        s = norm.final_stats["T"][k]
        assert s["mean"] == pytest.approx(atm["T"][..., k].mean(), rel=1e-9)
        assert s["std"] == pytest.approx(atm["T"][..., k].std(), rel=1e-9)  # population std


def test_bz_uses_final_b0(fitted):
    """Regression for commit 4cb32af: the stored Bz std belongs to asinh(Bz / B0_final).

    B0 (the asinh knee, in G) is the 60th percentile of |Bz| per node and is only known
    after all data are seen. Before 4cb32af the std was accumulated with a provisional B0,
    so it was about 3.5 times too large and the normalized Bz was squeezed toward 0. The
    test checks B0 against the percentile (5 %), the std against the RMS of
    asinh(Bz / B0) (2 %), and that the mean is fixed to 0 (Bz is not centered).
    """
    norm, atm = fitted
    for k in range(N_TAU):
        b0 = norm.B0_transform_per_tau[k]
        assert b0 == pytest.approx(np.percentile(np.abs(atm["Bz"][..., k]), 60), rel=0.05)
        y = np.arcsinh(atm["Bz"][..., k] / b0)
        rms = np.sqrt(np.mean(y ** 2))
        assert norm.final_stats["Bz"][k]["std"] == pytest.approx(rms, rel=0.02)
        assert norm.final_stats["Bz"][k]["mean"] == 0.0


@pytest.mark.parametrize("param", ["T", "Bz"])
def test_round_trip_is_identity(fitted, param):
    """``denormalize(transform(x))`` gives back T (K) and Bz (G) for every node.

    Every prediction returns to physical units through ``denormalize``, so any mismatch
    between the two directions would bias every reported value. T uses a z-score and Bz
    the asinh transform, so both code paths are covered.
    """
    norm, atm = fitted
    normalized = norm.transform(atm)[param]
    flat = normalized.reshape(-1, N_TAU)
    back = norm.denormalize(flat, param)
    np.testing.assert_allclose(back, atm[param].reshape(-1, N_TAU), rtol=1e-4, atol=1e-2)


@pytest.mark.xfail(
    reason="Known inconsistency: 'centered' transform divides by std without subtracting the "
    "mean, but denormalize adds the mean back, shifting Vz by +mean(tau). To be fixed in the "
    "reframe plan, Phase 1.",
    strict=True,
)
def test_vz_round_trip_is_identity(fitted):
    """Same round trip for Vz (km/s). It fails today on purpose (strict xfail).

    The "centered" transform divides by the std without subtracting the mean, but
    ``denormalize`` adds the mean back, so the round trip shifts Vz by +mean(tau)
    (at most 0.03 km/s on the paper nodes, 0.5 to 0.7 km/s at log tau = 0.8 to 1.4).
    Because ``strict=True``, the test turns into a failure once the bug is fixed, as a
    reminder to remove the marker.
    """
    norm, atm = fitted
    flat = norm.transform(atm)["Vz"].reshape(-1, N_TAU)
    back = norm.denormalize(flat, "Vz")
    np.testing.assert_allclose(back, atm["Vz"].reshape(-1, N_TAU), rtol=1e-4, atol=1e-3)


def test_tau_levels_are_independent(tmp_path):
    """Changing the data at one tau node does not change another node's statistics.

    Each of the optical-depth nodes is normalized separately. Shifting T by 1000 K at
    node 0 must change node 0's mean and leave node 1's stats bit-for-bit identical, so
    no statistic is accidentally pooled across heights.
    """
    atm_a = _atmosphere(seed=1)
    atm_b = {k: v.copy() for k, v in atm_a.items()}
    atm_b["T"][..., 0] += 1000.0
    stats = []
    for atm in (atm_a, atm_b):
        n = MhdNormalizer(n_tau=N_TAU, raw_values_dir=str(tmp_path))
        n.update(atm)
        stats.append(n.finalize())
    assert stats[0]["T"][1] == stats[1]["T"][1]
    assert stats[0]["T"][0]["mean"] != stats[1]["T"][0]["mean"]


def test_incremental_update_equals_single_pass(tmp_path):
    """Two ``update`` calls give the same stats as one call on the concatenated data.

    Training statistics are accumulated over steps 110 to 140 one step at a time, so the
    result must not depend on how the data are split between calls.
    """
    a, b = _atmosphere(seed=2), _atmosphere(seed=3)
    inc = MhdNormalizer(n_tau=N_TAU, raw_values_dir=str(tmp_path))
    inc.update(a)
    inc.update(b)
    inc.finalize()
    both = {k: np.concatenate([a[k], b[k]], axis=0) for k in a}
    for k in range(N_TAU):
        assert inc.final_stats["T"][k]["mean"] == pytest.approx(both["T"][..., k].mean(), rel=1e-9)
        assert inc.final_stats["T"][k]["std"] == pytest.approx(both["T"][..., k].std(), rel=1e-9)


def test_save_load_round_trip(fitted, tmp_path):
    """Stats and B0 survive JSON save/load, and the reloaded normalizer transforms identically.

    Training writes the JSON once and every analysis script reloads it. A lost key or a
    rounded float would silently change the normalization between training and inference.
    """
    norm, atm = fitted
    path = tmp_path / "mhd.json"
    norm.save(str(path), logtau_values=[-2.0, -1.0, 0.0])
    loaded = MhdNormalizer(n_tau=N_TAU, raw_values_dir=str(tmp_path)).load(str(path))
    assert loaded.final_stats == norm.final_stats
    np.testing.assert_allclose(loaded.B0_transform_per_tau, norm.B0_transform_per_tau)
    np.testing.assert_allclose(loaded.transform(atm)["Bz"], norm.transform(atm)["Bz"])


def test_saved_statistics_record_the_velocity_convention(fitted, tmp_path):
    """The MHD JSON stores which velocity its Vz statistics describe; older files load as None.

    Regenerated steps carry v_los = -Vz. Training compares this record with the data's
    convention (``check_normalizer_velocity_target``), so statistics of +Vz cannot normalize
    v_los with a mean of the wrong sign.
    """
    norm, _ = fitted
    norm.save(str(tmp_path / "new.json"), velocity_target="v_los")
    norm.save(str(tmp_path / "old.json"))
    assert MhdNormalizer(n_tau=N_TAU).load(str(tmp_path / "new.json")).velocity_target == "v_los"
    assert MhdNormalizer(n_tau=N_TAU).load(str(tmp_path / "old.json")).velocity_target is None


# --- StokesNormalizer -------------------------------------------------------------------

def _stokes(seed: int = 0) -> dict:
    """Small synthetic Stokes dict: I around 0.9 (continuum units) and V around 0 with sigma 1e-3."""
    rng = np.random.default_rng(seed)
    return {"I": 0.9 + 0.1 * rng.standard_normal((5, 4, 7)), "V": 1e-3 * rng.standard_normal((5, 4, 7))}


def test_stokes_stats_and_round_trip(tmp_path):
    """StokesNormalizer: streamed stats match numpy, the round trip is exact, save/load keeps stats.

    Checks, for I and V, that the mean and std accumulated over two updates equal numpy's,
    that ``transform`` gives order-one values, that ``inverse_transform`` recovers the
    input, and that the saved JSON reloads to the same statistics. The V tolerance has an
    absolute term because ``transform`` adds 1e-8 to the std and the inverse does not.
    """
    a, b = _stokes(0), _stokes(1)
    norm = StokesNormalizer()
    norm.update(a)
    norm.update(b)
    norm.finalize()
    for key in ("I", "V"):
        both = np.concatenate([a[key], b[key]]).ravel()
        assert norm.final_stats[key]["mean"] == pytest.approx(both.mean(), rel=1e-9, abs=1e-15)
        assert norm.final_stats[key]["std"] == pytest.approx(both.std(), rel=1e-9)
    t = norm.transform(a)
    assert abs(np.concatenate([t["I"].ravel()])).max() < 10
    flat = np.concatenate([t["I"].reshape(-1, 7), t["V"].reshape(-1, 7)], axis=1)
    back = norm.inverse_transform(flat)
    np.testing.assert_allclose(back["I"], a["I"].reshape(-1, 7), rtol=1e-6)
    # transform divides by (std + 1e-8) and inverse_transform multiplies by std only, so
    # the round trip carries a relative error of ~1e-8 / std (~1e-5 for V); hence atol.
    np.testing.assert_allclose(back["V"], a["V"].reshape(-1, 7), rtol=1e-4, atol=1e-7)

    path = tmp_path / "stokes.json"
    norm.save(path)
    loaded = StokesNormalizer().load(path)
    assert loaded.final_stats == norm.final_stats


# --- Target for reframe Phase 1c: Q and U as inputs ---------------------------------------

IQUV_TARGET = ("Agreed in the reframe plan (Phase 1c): Stokes Q and U become model inputs, so the "
               "Stokes normalizer must handle them. Today it only knows I and V.")


@pytest.mark.xfail(strict=True, reason=IQUV_TARGET)
def test_stokes_normalizer_handles_q_and_u():
    """StokesNormalizer accumulates, applies and inverts statistics for all four Stokes parameters.

    The polarization channels Q, U and V are divided by their standard deviation without
    subtracting their mean (their mean is a physical zero, not an offset), as the paper
    states; I is a z-score. The streamed statistics must match numpy and the round trip
    must recover the input.
    """
    rng = np.random.default_rng(0)
    data = {"I": 0.9 + 0.1 * rng.standard_normal((5, 4, 7)), "Q": 1e-3 * rng.standard_normal((5, 4, 7)),
            "U": 2e-3 * rng.standard_normal((5, 4, 7)), "V": 1e-3 * rng.standard_normal((5, 4, 7))}
    norm = StokesNormalizer()
    norm.update(data)
    norm.finalize()
    for key in ("Q", "U"):
        assert norm.final_stats[key]["std"] == pytest.approx(data[key].std(), rel=1e-9)
    t = norm.transform(data)
    assert set(t) >= {"I", "Q", "U", "V"}
    for key in ("Q", "U", "V"):
        np.testing.assert_allclose(t[key], data[key] / norm.final_stats[key]["std"], rtol=1e-5)
    flat = np.concatenate([t[k].reshape(-1, 7) for k in ("I", "Q", "U", "V")], axis=1)
    back = norm.inverse_transform(flat)
    for key in ("Q", "U"):
        np.testing.assert_allclose(back[key], data[key].reshape(-1, 7), rtol=1e-4, atol=1e-7)


def test_stokes_statistics_record_the_degradation_of_their_profiles(tmp_path):
    """The Stokes JSON stores the PSF settings of its profiles; reloading gives back only I and V stats.

    Training refuses statistics of differently degraded profiles
    (``check_stokes_normalizer_degradation``), so the record must survive the round trip,
    and files without it (before Phase 1b) must still load with ``degradation`` None.
    """
    norm = StokesNormalizer()
    norm.update(_stokes())
    norm.finalize()
    record = {"spatial_psf": True, "psf_path": "hinode-MODEST/PSFs/hinode_psf_bin.0.16.fits"}
    norm.save(str(tmp_path / "new.json"), degradation=record)
    norm.save(str(tmp_path / "old.json"))
    new = StokesNormalizer().load(str(tmp_path / "new.json"))
    assert new.degradation == record and set(new.final_stats) == {"I", "V"}
    assert StokesNormalizer().load(str(tmp_path / "old.json")).degradation is None
