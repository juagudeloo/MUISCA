"""Tests for the MUISCA-NICOLE bridge (utils/synthesis.py) and pixel sampling (utils/pixel_sampling.py).

The NICOLE tests run the real NICOLE v16.06 binary on a tiny HSRA atmosphere (under a
second) and skip when NICOLE or Python 2 is absent. They fix the sign conventions every
synthesized training profile depends on.
"""
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from utils.hinode_wavelengths import hinode_wavelength_grid
from utils.pixel_sampling import PixelSamplingResult, mark_overlay_subset, write_pixel_selection_outputs
from utils.synthesis import NicoleRunner, SynthesisComparator, SynthesisConfig

ROOT = Path(__file__).resolve().parent.parent
HSRA = ROOT / "data" / "csv" / "hsra_pgas.model"
LOGTAU = np.round(np.arange(-3.0, 1.45, 0.1), 2)        # the training grid, 45 nodes
C_KMS = 299792.458


def _nicole_available() -> bool:
    """True if the NICOLE binary, its driver and Python 2 are all present."""
    cfg_root = Path("/scratchsan/observatorio/juagudeloo/NICOLE_v16.06")
    return ((cfg_root / "main" / "nicole").exists() and (cfg_root / "run" / "run_nicole.py").exists()
            and shutil.which("python2") is not None and HSRA.exists())


needs_nicole = pytest.mark.skipif(not _nicole_available(), reason="NICOLE, Python 2 or the HSRA file absent")


def _config(tmp_path, **kwargs) -> SynthesisConfig:
    """MURaM-source configuration writing under tmp_path."""
    return SynthesisConfig(source="muram", experiment_root="test", model_type="base", muram_step=198,
                           output_root=Path(tmp_path), **kwargs)


def _hsra_temperature() -> np.ndarray:
    """HSRA temperature (K) on the 45 training nodes."""
    a = np.loadtxt(HSRA, skiprows=2)
    return np.interp(LOGTAU, a[:, 0], a[:, 1])


def _wfa_gauss(I: np.ndarray, V: np.ndarray, wl: np.ndarray) -> float:
    """Least-squares weak-field B (G) from Fe I 6302.5 A, positive toward the observer."""
    win = (wl > 6302.2) & (wl < 6302.8)
    dI = np.gradient(I, wl)[win]
    return float(-(V[win] * dI).sum() / (4.67e-13 * 6302.4936 ** 2 * 2.5 * (dI ** 2).sum()))


# --- NICOLE conventions ------------------------------------------------------------------

@needs_nicole
@pytest.mark.data
def test_nicole_signs_and_pixel_order(tmp_path):
    """NICOLE's v_LOS > 0 is a redshift and its B_long > 0 gives a positive WFA field.

    A 2 x 2 cube of the HSRA atmosphere has one static pixel, one with v_LOS = +2 km/s
    (expected core shift +42 mA), one with B_long = +500 G and one with -500 G, at known
    positions. The test checks each pixel's signature at its own position (so the
    cube's pixel order is preserved), the signs, and that Q and U are zero without a
    transverse field. The MURaM training synthesis relies on all of these: MURaM's Vz
    is positive upward, so it must enter NICOLE as -Vz.
    """
    runner = NicoleRunner(_config(tmp_path))
    T = np.broadcast_to(_hsra_temperature(), (2, 2, LOGTAU.size)).copy()
    v = np.zeros_like(T)
    b = np.zeros_like(T)
    v[0, 1] = 2.0
    b[1, 0] = 500.0
    b[1, 1] = -500.0
    iquv = runner.run_cube(tmp_path / "cube", LOGTAU, T, v, b, 112)
    wl = hinode_wavelength_grid()
    win = (wl > 6302.2) & (wl < 6302.8)

    def core(ix, iy):
        """Line-core wavelength (A) of Fe I 6302.5 from a parabola through the minimum."""
        I = iquv[ix, iy, 0, win]
        w = wl[win]
        j = int(np.clip(np.argmin(I), 1, I.size - 2))
        return w[j] + 0.5 * (I[j - 1] - I[j + 1]) / (I[j - 1] - 2 * I[j] + I[j + 1]) * (w[1] - w[0])

    shift_kms = (core(0, 1) - core(0, 0)) / 6302.4936 * C_KMS
    assert shift_kms == pytest.approx(2.0, abs=0.15)
    b_pos = _wfa_gauss(iquv[1, 0, 0], iquv[1, 0, 3], wl)
    b_neg = _wfa_gauss(iquv[1, 1, 0], iquv[1, 1, 3], wl)
    assert 350 < b_pos < 700 and -700 < b_neg < -350
    assert np.abs(iquv[0, 0, 3]).max() < 1e-6                         # no field, no V
    assert np.abs(iquv[..., 1:3, :]).max() < 1e-12                    # no transverse field, no Q, U


# --- Configuration -----------------------------------------------------------------------

def test_default_wavelength_grid_is_the_observed_one(tmp_path):
    """The default NICOLE grid is the MODEST FITS grid (first 6300.8736 A, 21.477 mA steps).

    The old hardcoded 6300.796 A start put every synthesized profile 0.0776 A (3.7 km/s)
    off the observations.
    """
    try:
        wl = hinode_wavelength_grid()
    except FileNotFoundError:
        pytest.skip("MODEST FITS header absent")
    cfg = _config(tmp_path)
    assert cfg.wl_first == pytest.approx(wl[0], abs=1e-9)
    assert cfg.wl_step_mA == pytest.approx((wl[1] - wl[0]) * 1e3, rel=1e-9)


def test_gt_pressure_runs_share_the_plain_runs_pixels(tmp_path):
    """A run with the MURaM gas pressure writes to its own folder but reads the plain run's pixel selection."""
    plain = _config(tmp_path, wl_first=6300.8736, wl_step_mA=21.477)
    gt = _config(tmp_path, wl_first=6300.8736, wl_step_mA=21.477, add_gt_pressure=True)
    assert gt.out_dir() != plain.out_dir()
    assert gt.sampling_dir() == plain.out_dir()


# --- Chi-square --------------------------------------------------------------------------

def test_chi_square_compares_i_with_i_and_v_with_v(tmp_path):
    """chi2 uses channel 0 (I) and 1 (V) of the input and channel 0 (I) and 3 (V) of the synthesis.

    The input file stores (I, V) and the synthesis (I, Q, U, V), so mixing the indices
    would compare V with Q. With synthesized I off by 1e-3 everywhere and V exact, and
    sigma = 1e-3, chi2_I must equal the number of wavelengths and chi2_V must be 0.
    """
    cfg = _config(tmp_path, wl_first=6300.8736, wl_step_mA=21.477)
    nwl = 7
    obs = np.zeros((1, 2, nwl)); obs[0, 0] = 1.0; obs[0, 1] = 0.01
    syn = np.zeros((1, 4, nwl)); syn[0, 0] = 1.001; syn[0, 1] = 0.5; syn[0, 3] = 0.01
    pred, synth = tmp_path / "p.h5", tmp_path / "s.h5"
    with h5py.File(pred, "w") as h:
        h["pixels"] = np.array([[3, 4]]); h["stokes_obs"] = obs; h["wavelengths"] = np.arange(nwl)
        h["Bz"] = np.zeros((1, 3))
    with h5py.File(synth, "w") as h:
        h["pixels"] = np.array([[3, 4]]); h["stokes_synth"] = syn
    comp = SynthesisComparator(cfg)
    comp.load(pred, synth)
    chi = comp.chi_square(sigma_I=1e-3, sigma_V=1e-3)["3,4"]
    assert chi["chi2_I"] == pytest.approx(nwl, rel=1e-6)
    assert chi["chi2_V"] == pytest.approx(0.0, abs=1e-9)


# --- Pixel sampling ----------------------------------------------------------------------

def _result() -> PixelSamplingResult:
    """Selection of 5 pixels in 2 bins on a 3 x 3 map."""
    pixels = [(0, 0), (0, 1), (0, 2), (2, 0), (2, 2)]
    bins = {(0, 0): 0, (0, 1): 0, (0, 2): 0, (2, 0): 1, (2, 2): 1}
    return PixelSamplingResult(pixels=pixels, bin_of_pixel=bins, bin_edges=np.array([1.0, 10.0, 1000.0]),
                               bin_values=[2.0, 3.0, 4.0, 300.0, 900.0], tau_index=2, logtau_value=0.0,
                               abs_bz_map=np.geomspace(1, 1000, 9).reshape(3, 3), pred_nx=3, pred_ny=3)


def test_overlay_subset_is_drawn_within_each_bin():
    """At most ``n_overlay_per_bin`` pixels per bin are marked, and only selected pixels are marked."""
    r = _result()
    marks = mark_overlay_subset(r, n_overlay_per_bin=2, seed=1)
    assert set(marks) == set(r.pixels)
    per_bin = [sum(marks[p] for p in r.pixels if r.bin_of_pixel[p] == b) for b in (0, 1)]
    assert per_bin == [2, 2]


def test_selection_json_records_pixels_bins_and_fields(tmp_path):
    """The saved JSON lists every pixel with its bin and |B_LOS| (G), plus the bin edges."""
    import json
    r = _result()
    paths = write_pixel_selection_outputs(r, tmp_path, "exp", "base", is_overlay=mark_overlay_subset(r, 1))
    data = json.loads(paths["json"].read_text())
    assert data["bin_edges_gauss"] == [1.0, 10.0, 1000.0]
    assert [(p["ix"], p["iy"], p["bin"], p["abs_bz_gauss"]) for p in data["pixels"]] == \
        [(ix, iy, r.bin_of_pixel[(ix, iy)], v) for (ix, iy), v in zip(r.pixels, r.bin_values)]
    assert data["n_overlay_pixels"] == 2 and paths["png"].exists()


@needs_nicole
@pytest.mark.data
def test_nicole_transverse_field_convention(tmp_path):
    """NICOLE's Q core goes as -cos(2 chi) and U core as -sin(2 chi), chi the field azimuth from +x.

    An 800 G horizontal field along x gives Q < 0, along y gives Q > 0, and at 45 degrees
    gives U < 0 with Q about 0. Raw Hinode Q and U follow the same form against SPINOR's
    azimuth (r = -0.58 and -0.73), so MODEST's Q and U are loaded without a sign flip.
    Uses the single-pixel ASCII model, which already writes Bx and By.
    """
    from utils.model_prof_tools import write_ascii_model
    cfg = _config(tmp_path)
    runner = NicoleRunner(cfg)
    T = _hsra_temperature()
    zero = np.zeros_like(T)
    core = int(np.argmin(np.abs(hinode_wavelength_grid() - 6302.4936)))
    out = {}
    for name, bx, by in (("x", 800.0, 0.0), ("y", 0.0, 800.0), ("45", 566.0, 566.0)):
        wd = runner.prepare_workdir(1, 1, LOGTAU, T, zero, zero)
        write_ascii_model(wd / "model.model", logtau=LOGTAU, T=T, v_los_kms=zero, b_long_G=zero,
                          v_mic_cms=cfg.v_mic_cms, el_p_seed=cfg.el_p_seed,
                          b_x_G=np.full_like(T, bx), b_y_G=np.full_like(T, by))
        out[name] = runner.run_pixel(wd)[:, core]
    assert out["x"][1] < -0.01 and out["y"][1] > 0.01
    assert out["45"][2] < -0.01 and abs(out["45"][1]) < 1e-3


@needs_nicole
@pytest.mark.data
def test_run_cube_synthesizes_q_and_u_like_the_single_pixel_path(tmp_path):
    """A cube with a horizontal field gives the Q and U of the single-pixel ASCII path.

    The two paths already differ by about 3e-3 (continuum units) in I for the same atmosphere
    (NICOLE treats an isolated pixel slightly differently, see ``run_cube``), so Q must agree
    to that level and correlate above 0.9999, and U must stay zero for a field along x. A
    wrong transverse slot would give Q = 0, a sign flip or Q leaking into U.
    """
    from utils.model_prof_tools import write_ascii_model
    cfg = _config(tmp_path)
    runner = NicoleRunner(cfg)
    T = _hsra_temperature()
    zero = np.zeros_like(T)
    Tc = np.broadcast_to(T, (1, 2, T.size)).copy()
    bx = np.zeros_like(Tc); bx[0, 1] = 800.0
    cube = runner.run_cube(tmp_path / "cube", LOGTAU, Tc, np.zeros_like(Tc), np.zeros_like(Tc), 112,
                           b_x_G=bx, b_y_G=np.zeros_like(Tc))
    wd = runner.prepare_workdir(1, 1, LOGTAU, T, zero, zero)
    write_ascii_model(wd / "model.model", logtau=LOGTAU, T=T, v_los_kms=zero, b_long_G=zero,
                      v_mic_cms=cfg.v_mic_cms, el_p_seed=cfg.el_p_seed, b_x_G=np.full_like(T, 800.0), b_y_G=zero)
    single = runner.run_pixel(wd)
    np.testing.assert_allclose(cube[0, 1, 1], single[1], atol=3e-3)
    assert np.corrcoef(cube[0, 1, 1], single[1])[0, 1] > 0.9999
    assert np.abs(cube[0, 1, 2]).max() < 1e-6 and np.abs(cube[0, 1, 1]).max() > 0.01


@needs_nicole
@pytest.mark.data
def test_transverse_field_floor_grows_with_the_horizontal_field(tmp_path):
    """Leaving out Bx and By costs no chi^2 without a horizontal field and more for a stronger one.

    The same HSRA atmosphere (B_LOS = 300 G) with a horizontal field of 0, 200 and 800 G is
    synthesized with and without that field; chi^2_I and chi^2_V between the two (sigma 1e-3)
    must be 0 for 0 G and grow with the field. This is the floor that the re-synthesis of
    MUISCA's predictions, which have no transverse field, inherits.

    The three columns are 0, 5 and 10 K warmer than the HSRA (the same in both runs): NICOLE
    is off by about 3e-3 for a pixel identical to its predecessor, which would give chi^2 of
    about 1e3 in the run without the field and make the test pass for the wrong reason.
    """
    from scripts.synthesis.transverse_field_floor import chi2_between
    runner = NicoleRunner(_config(tmp_path))
    T = np.broadcast_to(_hsra_temperature(), (1, 3, LOGTAU.size)).copy()
    T += np.array([0.0, 5.0, 10.0])[None, :, None]
    b = np.full_like(T, 300.0)
    bx = np.zeros_like(T); bx[0, 1] = 200.0; bx[0, 2] = 800.0
    with_t = runner.run_cube(tmp_path / "w", LOGTAU, T, np.zeros_like(T), b, 112, b_x_G=bx, b_y_G=np.zeros_like(T))[0]
    without = runner.run_cube(tmp_path / "wo", LOGTAU, T, np.zeros_like(T), b, 112)[0]
    chi_i = chi2_between(with_t[:, 0], without[:, 0])
    chi_v = chi2_between(with_t[:, 3], without[:, 3])
    assert chi_i[0] < 1e-6 and chi_v[0] < 1e-6
    assert chi_i[2] > chi_i[1] > 0 and chi_v[2] > chi_v[1] > 0
