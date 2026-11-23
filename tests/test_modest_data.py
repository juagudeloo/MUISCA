"""Tests for utils/modest_data.py: MODEST preprocessing, cropping and SPINOR plane mapping.

Synthetic tests use small arrays. The ``data`` tests read the MODEST scan in data/ and skip
when it is absent.
"""
from pathlib import Path

import numpy as np
import pytest
from scipy.fft import fft2, ifft2

from utils.modest_data import ModestData, transform_modest_stokes_profiles

ROOT = Path(__file__).resolve().parent.parent
MODEST_DIR = ROOT / "data" / "hinode-MODEST" / "INV_560_AR11967"
needs_modest = pytest.mark.skipif(not (MODEST_DIR / "inverted_obs.1.fits").exists(),
                                  reason="MODEST scan absent")


def _gaussian_psf(size: int = 9, sigma: float = 1.2) -> np.ndarray:
    """Centred 2-D Gaussian PSF normalized to sum 1."""
    y, x = np.mgrid[:size, :size] - size // 2
    psf = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return psf / psf.sum()


def _periodic_blur(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Convolve ``image`` with ``psf`` periodically, the PSF centred on pixel (0, 0)."""
    pad = np.zeros_like(image)
    k = psf.shape[0]
    pad[:k, :k] = psf
    pad = np.roll(pad, (-(k // 2), -(k // 2)), axis=(0, 1))
    return np.real(ifft2(fft2(image) * fft2(pad)))


# --- transform_modest_stokes_profiles ------------------------------------------------------

def test_transform_defaults_are_identity():
    """With default arguments the transform returns the input unchanged (I and V, float32)."""
    rng = np.random.default_rng(0)
    stokes = {"I": rng.random((2, 3, 7)), "V": rng.standard_normal((2, 3, 7))}
    out = transform_modest_stokes_profiles(stokes, np.arange(7.0))
    for k in ("I", "V"):
        np.testing.assert_allclose(out[k], stokes[k].astype(np.float32))


def test_transform_shift_moves_a_line_by_the_requested_samples():
    """A shift of +2 samples moves a line core two samples redward, i.e. about +2.04 km/s on Hinode.

    One Hinode sample is 21.5 mA, so a sign error in the shift would bias every velocity
    the model reads from MODEST. Scaling and reversing are checked on the same profile.
    """
    wl = np.arange(20.0)
    line = 1 - 0.5 * np.exp(-((wl - 8.0) ** 2) / 4.0)
    stokes = {"I": line[None, None], "V": np.zeros((1, 1, 20))}
    shifted = transform_modest_stokes_profiles(stokes, wl, shift_positions=2.0)["I"][0, 0]
    assert int(np.argmin(shifted)) == 10
    scaled = transform_modest_stokes_profiles(stokes, wl, i_scale=0.5)["I"][0, 0]
    np.testing.assert_allclose(scaled, 0.5 * line, rtol=1e-6)
    flipped = transform_modest_stokes_profiles(stokes, wl, invert_direction=True)["I"][0, 0]
    np.testing.assert_allclose(flipped, line[::-1], rtol=1e-6)


# --- Deconvolution -----------------------------------------------------------------------

def test_wiener_deconvolution_undoes_a_known_blur_without_shifting():
    """Deconvolving a periodically blurred map with a small regularization recovers the map.

    This checks that the PSF is placed at the origin correctly: a half-pixel or one-pixel
    misplacement would shift every deconvolved map and fail the comparison.
    """
    rng = np.random.default_rng(0)
    image = 1.0 + 0.1 * rng.standard_normal((32, 32))
    psf = _gaussian_psf(sigma=0.8)          # min |P|^2 about 3e-6, well above the regularization
    out = ModestData._wiener_deconv(_periodic_blur(image, psf), psf, 1e-10)
    np.testing.assert_allclose(out, image, atol=1e-3)


@pytest.mark.xfail(strict=True, reason="Known: the Wiener filter has gain 1 / (1 + noise_level) at zero "
                   "frequency, so the deconvolved continuum is 1 % too dark with the default 0.01. The "
                   "reframe plan stops using this path (native input, Phase 1d).")
def test_wiener_deconvolution_preserves_the_mean():
    """Deconvolution must not change the mean intensity of a map (here a flat continuum of 1)."""
    out = ModestData._wiener_deconv(np.ones((16, 16)), _gaussian_psf(), 0.01)
    assert out.mean() == pytest.approx(1.0, rel=1e-4)


# --- Grids, masks and regions -------------------------------------------------------------

def test_downsample_is_an_exact_block_mean():
    """Block-averaging by 2 gives the mean of each 2 x 2 block, ignores NaN, and rejects non-integer factors."""
    cube = np.arange(32.0).reshape(4, 4, 2)
    cube[0, 0, 0] = np.nan
    out = ModestData._downsample_cube_to_shape(cube, (2, 2))
    assert out[0, 0, 0] == pytest.approx(np.nanmean(cube[:2, :2, 0]))
    assert out[1, 1, 1] == pytest.approx(cube[2:, 2:, 1].mean())
    with pytest.raises(ValueError):
        ModestData._downsample_cube_to_shape(cube, (3, 3))


def test_region_crops_cover_the_same_area_on_every_grid():
    """A crop on the observed grid selects the same sky area on the upsampled grid.

    Each pixel stores its own observed-grid (y, x) code, upsampled by repetition, so the
    two crops must hold exactly the same codes. This guards against the regions of the
    deconvolved input and of the SPINOR maps drifting apart.
    """
    y, x = np.mgrid[:6, :10]
    code = (100 * y + x).astype(float)
    m = ModestData()
    m.continuum = code
    m.obs_stokes = {"I": code[..., None]}
    m.smoothed_stokes = {"I": np.repeat(np.repeat(code, 2, 0), 2, 1)[..., None]}
    r = m.extract_region(1, 4, 3, 8)
    coarse = r["obs_stokes"]["I"][..., 0]
    fine = r["smoothed_stokes"]["I"][..., 0]
    np.testing.assert_array_equal(fine[::2, ::2], coarse)
    np.testing.assert_array_equal(r["continuum"], coarse)


def test_mask_keeps_weakly_polarized_pixels():
    """The mask is True where mean |V| / I is at most the threshold, and masking sets the others to NaN."""
    m = ModestData(circular_polarization_threshold=0.01)
    m.obs_stokes = {"I": np.ones((1, 3, 4)), "V": np.array([[[0.0] * 4, [0.005] * 4, [0.05] * 4]])}
    m.apply_mask_to_data()
    assert m.mask.tolist() == [[True, True, False]]
    assert np.isnan(m.obs_stokes["V"][0, 2]).all() and not np.isnan(m.obs_stokes["V"][0, :2]).any()


def test_spinor_planes_map_to_the_right_quantities():
    """Each SPINOR quantity and node reads its own plane of inverted_atmos.fits (0-based).

    Every plane holds its own index, so a wrong default (e.g. reading the next parameter
    block) shows up as a wrong number. B_LOS is B cos(gamma), here gamma = 0.
    """
    m = ModestData()
    m.inverted_atmos = np.arange(23, dtype=np.float32)[:, None, None] * np.ones((1, 1, 1), np.float32)
    m.inverted_atmos[11:14] = 0.0                      # gamma = 0 degrees
    atm = m.build_spinor_atmosphere()
    assert [atm["T"][t][0, 0] for t in (-2.0, -0.8, 0.0)] == [5, 6, 7]
    assert [atm["Vlos"][t][0, 0] for t in (-2.0, -0.8, 0.0)] == [17, 18, 19]
    assert [atm["Blos"][t][0, 0] for t in (-2.0, -0.8, 0.0)] == [8, 9, 10]


# --- Real data: channel order and sign conventions ------------------------------------------

def _wfa(I: np.ndarray, V: np.ndarray, wl: np.ndarray) -> np.ndarray:
    """Least-squares weak-field B (G, positive toward the observer) from Fe I 6302.5 A."""
    dI = np.gradient(I, wl, axis=-1)
    c = 4.67e-13 * 6302.4936 ** 2 * 2.5
    return -(V * dI).sum(-1) / (c * (dI ** 2).sum(-1))


@needs_modest
@pytest.mark.data
def test_observed_stokes_channel_order_matches_spinor_fit():
    """Channel 1 of inverted_obs is V: it correlates with SPINOR's fitted V far better than Q or U do."""
    from astropy.io import fits
    with fits.open(MODEST_DIR / "inverted_obs.1.fits") as h:
        obs = np.asarray(h[0].data[150:250, 300:400], dtype=np.float64)
    with fits.open(MODEST_DIR / "inverted_profs.1.fits") as h:
        fit_v = np.asarray(h[0].data[150:250, 300:400, 1], dtype=np.float64)
    obs_amp = np.abs(obs).max(-1)               # (ny, nx, 4) peak amplitude per channel
    fit_amp = np.abs(fit_v).max(-1)
    corr = [np.corrcoef(obs_amp[..., k].ravel(), fit_amp.ravel())[0, 1] for k in range(4)]
    assert int(np.argmax(corr[1:])) + 1 == 1 and corr[1] > 0.9


@needs_modest
@pytest.mark.data
@pytest.mark.xfail(strict=True, reason="Known, fixed in reframe Phase 1d: the default stokes_v_multiplier = -1 "
                   "flips Hinode V, whose raw sign already matches NICOLE's. The paper tables compensate with "
                   "--modest-pred-blos-invert-sign 1.")
def test_loaded_v_gives_spinor_polarity():
    """The V that ModestData hands to the model gives B_LOS with SPINOR's polarity under the WFA.

    MURaM-trained models map NICOLE's V to B with the WFA sign (corr +0.94 on step 110),
    so the loaded Hinode V must give a positive WFA correlation with SPINOR's
    B cos(gamma) at log tau = -0.8. Raw Hinode V gives +0.86.
    """
    from utils.hinode_wavelengths import hinode_wavelength_grid
    m = ModestData()
    stokes = m.load_obs_stokes()
    m.load_inverted_atmos()
    atm = m.build_spinor_atmosphere()
    wl = hinode_wavelength_grid()
    win = (wl > 6302.2) & (wl < 6302.8)
    b = _wfa(stokes["I"][..., win].astype(np.float64), stokes["V"][..., win].astype(np.float64), wl[win])
    assert np.corrcoef(b.ravel(), atm["Blos"][-0.8].ravel())[0, 1] > 0.5


# --- MODEST cache round trip ---------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason="Known, fixed in reframe Phase 1d: ModestDataCache stores the SPINOR "
                   "log tau keys as float32, so -0.8 comes back as -0.800000011920929 and lookups with the "
                   "literal -0.8 (e.g. plot_spinor_atmosphere's default) raise KeyError after a cache hit.")
def test_modest_cache_keeps_spinor_node_keys(tmp_path):
    """A cached SPINOR atmosphere reloads with exactly the same log tau keys (-2.0, -0.8, 0.0)."""
    import contextlib
    import io
    from utils.cache_manage import ModestDataCache
    cache = ModestDataCache(str(tmp_path))
    atm = {"T": {t: np.full((1, 1), 5000.0) for t in (-2.0, -0.8, 0.0)}}
    with contextlib.redirect_stdout(io.StringIO()):
        cache.save("h", {"spinor_atm": atm})
        back = cache.load("h")
    assert sorted(back["spinor_atm"]["T"]) == [-2.0, -0.8, 0.0]


# --- Targets for reframe Phase 1d: Q and U from MODEST ------------------------------------

IQUV_TARGET = ("Agreed in the reframe plan (Phase 1d): Stokes Q and U from MODEST become model inputs. "
               "Today the loader and the transform keep only I and V.")


@pytest.mark.xfail(strict=True, reason=IQUV_TARGET)
def test_transform_keeps_all_four_stokes_parameters():
    """The spectral transform shifts and returns Q and U like I and V (here a one-sample shift)."""
    wl = np.arange(6.0)
    base = np.arange(6.0).reshape(1, 1, 6)
    stokes = {k: base * (i + 1) for i, k in enumerate(("I", "Q", "U", "V"))}
    out = transform_modest_stokes_profiles(stokes, wl, shift_positions=1.0)
    assert set(out) == {"I", "Q", "U", "V"}
    np.testing.assert_allclose(out["U"][0, 0], 3 * (np.arange(6.0) - 1.0))


@needs_modest
@pytest.mark.data
@pytest.mark.xfail(strict=True, reason=IQUV_TARGET)
def test_loaded_q_and_u_follow_nicoles_convention():
    """Observed Q and U load with NICOLE's sign convention: Q core ~ -cos(2 chi), U core ~ -sin(2 chi).

    chi is SPINOR's azimuth at log tau = -0.8. NICOLE gives Q < 0 for a field along x and
    U < 0 at 45 degrees (``tests/test_synthesis.py``), and raw Hinode Q and U follow the
    same form, so they must be loaded unflipped. Only pixels with a transverse field above
    300 G are used, where Q and U are well above the noise.
    """
    from utils.hinode_wavelengths import hinode_wavelength_grid
    m = ModestData()
    stokes = m.load_obs_stokes()
    atmos = m.load_inverted_atmos()
    wl = hinode_wavelength_grid()
    core = int(np.argmin(np.abs(wl - 6302.4936)))
    b_perp = atmos[9] * np.sin(np.deg2rad(atmos[12]))
    chi = np.deg2rad(atmos[15])
    sel = b_perp > 300.0
    q, u = stokes["Q"][..., core][sel], stokes["U"][..., core][sel]
    assert np.corrcoef(q, -np.cos(2 * chi[sel]))[0, 1] > 0.3
    assert np.corrcoef(u, -np.sin(2 * chi[sel]))[0, 1] > 0.3


@needs_modest
@pytest.mark.data
@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1d): a 'native' input mode feeds the raw "
                   "observed I, Q, U, V (0.32 arcsec, no deconvolution or smoothing) times one quiet-Sun factor.")
def test_native_input_is_the_raw_cube_rescaled_to_the_quiet_sun():
    """In native mode the model input is the observed cube times ``qs_rescale_factor``, on the observed grid.

    The factor brings the MODEST quiet-Sun continuum to the MURaM quiet-Sun continuum in
    HSRA units. Measured on 2026-11-23 the two agree within 0.3 %, so the factor must be
    within 0.99 to 1.02, and the same factor applies to all four Stokes parameters.
    """
    m = ModestData(stokes_v_multiplier=1.0)
    data = m.load_all(apply_mask=False, use_cache=False, prediction_input_mode="native")
    raw = m.load_obs_stokes()
    f = m.qs_rescale_factor
    assert 0.99 < f < 1.02
    for k in ("I", "Q", "U", "V"):
        assert data["prediction_stokes"][k].shape == raw[k].shape
        np.testing.assert_allclose(data["prediction_stokes"][k], f * raw[k], rtol=1e-6)
