"""Cache-signature regressions: every setting that changes cached content must change the signature.

Training caches data in two layers (see utils/cache_manage.py). The per-step cache holds the
degraded Stokes profiles, the MHD targets and the physics approximations, all un-normalized.
The balanced cache holds normalized tensors and the approximations again. If a setting
changes what a layer stores but not that layer's signature, the cache silently serves stale
data. This happened before: the black-body approximation was changed and the balanced cache
kept serving the old temperatures.

Each test changes one setting and checks that the relevant signature changes. Settings that
are known to be missing from a signature are strict xfails, to be fixed when the signatures
are reworked in the reframe plan (Phase 1b).
"""
import dataclasses

import pytest

from scripts.base_training import (
    TrainingConfig,
    build_balanced_cache_signature,
    build_cache_config_signature,
)
from utils.cache_manage import BalancedTrainDataCache, MuramDataCache

STEPS = [110, 120, 130, 140]
GAP = "Known gap, fixed in reframe Phase 1b: this setting changes the cached content but not this signature."


def _variant(**changes) -> TrainingConfig:
    """Default TrainingConfig with some fields replaced."""
    return dataclasses.replace(TrainingConfig(), **changes)


def _raw_hash(config: TrainingConfig) -> str:
    """Per-step cache hash, as the training script computes it."""
    return MuramDataCache.make_config_hash(build_cache_config_signature(config))


def _balanced_hash(config: TrainingConfig) -> str:
    """Balanced cache hash, as the training script computes it."""
    return BalancedTrainDataCache.make_signature_hash(build_balanced_cache_signature(config, STEPS))


# Settings that change the per-step cache: the degraded profiles or the approximations.
RAW_CASES = [
    pytest.param({"wl_range": (10, 60)}, id="wl_range (WFA and Doppler window)"),
    pytest.param({"central_wavelength": 6302.5}, id="central_wavelength (WFA and Doppler)"),
    pytest.param({"temp_continuum_indices": [0, 1, 2, 3]}, id="temp_continuum_indices (black body)"),
    pytest.param({"temp_continuum_wavelength": 6300.9}, id="temp_continuum_wavelength (black body)"),
    pytest.param({"stokes_cont_indices": [4, 5, 6, 7]}, id="stokes_cont_indices"),
    pytest.param({"lande_factor": 2.5}, id="lande_factor (WFA)",
                 marks=pytest.mark.xfail(strict=True, reason=GAP)),
    pytest.param({"lsf_path": "hinode-MODEST/PSFs/other.psf"}, id="lsf_path (spectral degradation)",
                 marks=pytest.mark.xfail(strict=True, reason=GAP)),
]

# Settings that change the balanced cache: the same approximations it copies, plus the
# normalization (it stores normalized tensors).
BALANCED_CASES = [
    pytest.param({"temp_continuum_indices": [0, 1, 2, 3]}, id="temp_continuum_indices (black body)"),
    pytest.param({"stokes_cont_indices": [4, 5, 6, 7]}, id="stokes_cont_indices"),
    pytest.param({"wl_range": (10, 60)}, id="wl_range (WFA and Doppler window)",
                 marks=pytest.mark.xfail(strict=True, reason=GAP)),
    pytest.param({"central_wavelength": 6302.5}, id="central_wavelength (WFA and Doppler)",
                 marks=pytest.mark.xfail(strict=True, reason=GAP)),
    pytest.param({"lande_factor": 2.5}, id="lande_factor (WFA)",
                 marks=pytest.mark.xfail(strict=True, reason=GAP)),
    pytest.param({"mhd_normalizer_path": "normalization_stats/other/mhd_normalization.json"},
                 id="mhd normalization stats", marks=pytest.mark.xfail(strict=True, reason=GAP)),
    pytest.param({"stokes_normalizer_path": "normalization_stats/other/stokes_normalization.json"},
                 id="stokes normalization stats", marks=pytest.mark.xfail(strict=True, reason=GAP)),
]


# Settings added by the reframe plan (Phase 1b, 1c): they change what both caches store.
NEW = "Agreed in the reframe plan: new setting, not in TrainingConfig yet."
PHASE1_CASES = [
    pytest.param({"spatial_psf": True}, id="spatial_psf (Phase 1b)", marks=pytest.mark.xfail(strict=True, reason=NEW)),
    pytest.param({"hinode_noise": 1e-3}, id="hinode_noise (Phase 1b)", marks=pytest.mark.xfail(strict=True, reason=NEW)),
    pytest.param({"stokes_channels": ("I", "V")}, id="stokes_channels (Phase 1c)",
                 marks=pytest.mark.xfail(strict=True, reason=NEW)),
]


@pytest.mark.parametrize("changes", PHASE1_CASES)
def test_new_settings_change_both_signatures(changes):
    """The PSF, noise and channel settings change both cache signatures (each changes the cached input)."""
    assert _raw_hash(_variant(**changes)) != _raw_hash(TrainingConfig())
    assert _balanced_hash(_variant(**changes)) != _balanced_hash(TrainingConfig())


@pytest.mark.parametrize("changes", RAW_CASES)
def test_setting_changes_the_per_step_cache_signature(changes):
    """Changing a setting that alters the per-step cache's content gives a new hash.

    Covers the window and wavelength of the WFA and Doppler approximations, the Lande
    factor, the black-body continuum points, the Stokes continuum points and the LSF file.
    """
    assert _raw_hash(_variant(**changes)) != _raw_hash(TrainingConfig())


@pytest.mark.parametrize("changes", BALANCED_CASES)
def test_setting_changes_the_balanced_cache_signature(changes):
    """Changing a setting that alters the balanced cache's content gives a new hash.

    The balanced cache copies the approximations and stores tensors normalized with the
    current statistics, so the approximation settings and the normalization statistics
    must all be part of its signature. Otherwise, recomputing the statistics (as the
    pending Bz-normalization retrain requires) would train on tensors normalized with the
    old ones.
    """
    assert _balanced_hash(_variant(**changes)) != _balanced_hash(TrainingConfig())


def test_signatures_ignore_settings_that_do_not_change_the_data():
    """Optimizer and logging settings must not invalidate either cache (a full rebuild takes hours)."""
    other = _variant(learning_rate=TrainingConfig().learning_rate * 10, n_epochs=TrainingConfig().n_epochs + 1)
    assert _raw_hash(other) == _raw_hash(TrainingConfig())
    assert _balanced_hash(other) == _balanced_hash(TrainingConfig())


def test_balanced_signature_changes_with_the_training_steps():
    """A different set of training steps gives a new balanced hash (it is cached per step set)."""
    base = build_balanced_cache_signature(TrainingConfig(), STEPS)
    other = build_balanced_cache_signature(TrainingConfig(), STEPS[:-1])
    assert BalancedTrainDataCache.make_signature_hash(base) != BalancedTrainDataCache.make_signature_hash(other)


def test_regenerated_source_files_change_both_signatures(tmp_path):
    """Rewriting a step's source files (same names, new content) changes both cache hashes.

    Regenerated steps keep their file names, so without the source fingerprint the caches
    would keep serving the old profiles and targets (e.g. the old velocity sign) after a
    regeneration.
    """
    sim = tmp_path / "muram-simulation"
    sim.mkdir()
    for name in ("stokes_110_nicole_tau500.npy", "atmos_110_tau500.npz"):
        (sim / name).write_bytes(b"old")
    cfg = _variant(data_path=str(tmp_path))
    raw0 = MuramDataCache.make_config_hash(build_cache_config_signature(cfg, step=110))
    bal0 = BalancedTrainDataCache.make_signature_hash(build_balanced_cache_signature(cfg, [110]))
    (sim / "atmos_110_tau500.npz").write_bytes(b"regenerated, longer")
    assert MuramDataCache.make_config_hash(build_cache_config_signature(cfg, step=110)) != raw0
    assert BalancedTrainDataCache.make_signature_hash(build_balanced_cache_signature(cfg, [110])) != bal0
