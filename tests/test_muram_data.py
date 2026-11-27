"""Tests for utils/muram_data.py: MURaM loading, optical depth, Hinode degradation and sampling.

The synthetic tests build tiny cubes in memory or in tmp_path. The ``data`` tests read the
Hinode LSF, the MODEST FITS header (wavelength grid) or a MURaM step, and skip when absent.
"""
import contextlib
import io
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from utils.muram_data import (
    MhdData,
    MuramStepDataset,
    StokesData,
    _otsu_threshold,
    build_balanced_region_indices,
    build_bz_strength_balanced_indices,
)
from utils.normalizer import MhdNormalizer, StokesNormalizer

ROOT = Path(__file__).resolve().parent.parent
LSF_PATH = ROOT / "data" / "hinode-MODEST" / "PSFs" / "hinode_sp.spline.psf"
MURAM_DIR = ROOT / "data" / "muram-simulation"
C_KMS = 299792.458


def _quiet(fn, *args, **kwargs):
    """Call ``fn`` with its progress prints suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _write_constant_kappa_table(path: Path, kappa_cm2_g: float) -> None:
    """Write a kappa.0.dat-style table (56 T x 21 P indices) with one constant opacity."""
    rows = [f"{it} {ip} {kappa_cm2_g:.6E}" for it in range(56) for ip in range(21)]
    path.write_text("\n".join(rows) + "\n")


# --- MhdData -----------------------------------------------------------------------------

def test_load_step_layout_and_units(tmp_path):
    """``load_step`` transposes the (nx, nz, ny) files to (nx, ny, nz) and converts units.

    A tiny fake snapshot (nx=2, ny=3, nz=4) stores each cell's own (x, y, z) code in the
    files, so a wrong transpose would put a cell's value at another position. Vz must be
    result_2 / result_0 / 1e5 (cm/s to km/s) and Bz must be result_6 times sqrt(4 pi)
    (code units to G).
    """
    nx, ny, nz = 2, 3, 4
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    code = (100 * x + 10 * y + z + 1).astype(np.float32)          # (nx, ny, nz), all > 0

    def to_file(arr_xyz):
        """Reorder an (nx, ny, nz) array to the (nx, nz, ny) file layout."""
        return np.ascontiguousarray(arr_xyz.transpose(0, 2, 1))   # file order (nx, nz, ny)

    rho = np.full((nx, ny, nz), 2.0, np.float32)
    np.stack([to_file(5000 + code), to_file(1e5 + code)]).tofile(tmp_path / "eos.110000")
    to_file(rho).tofile(tmp_path / "result_0.110000")
    to_file(rho * code * 1e5).tofile(tmp_path / "result_2.110000")   # rho * vz in cm/s
    to_file(code).tofile(tmp_path / "result_6.110000")

    mhd = MhdData(str(tmp_path), nx=nx, ny=ny, nz=nz)
    data = _quiet(mhd.load_step, 110)
    np.testing.assert_allclose(data["T"].to_value(u.K), 5000 + code)
    np.testing.assert_allclose(data["P"].to_value(u.dyn / u.cm ** 2), 1e5 + code)
    np.testing.assert_allclose(data["Vz"].to_value(u.km / u.s), code, rtol=1e-6)
    np.testing.assert_allclose(data["Bz"].to_value(u.G), code * np.sqrt(4 * np.pi), rtol=1e-6)

    trimmed = _quiet(MhdData(str(tmp_path), nx=nx, ny=ny, nz=nz).load_step, "110", z_max=2)
    assert trimmed["T"].shape == (nx, ny, 2)


def test_optical_depth_is_linear_for_constant_opacity(tmp_path):
    """With constant kappa and rho, tau grows linearly from the top: tau(z) = kappa rho dz (nz-1-z).

    kappa = 0.5 cm^2 g^-1 and rho = 1e-7 g cm^-3 give kappa rho = 5e-8 cm^-1, so each
    10 km (1e6 cm) layer adds 0.05 to tau. The top layer has tau = 0, clipped to 1e-30
    (log tau = -30). This checks the integration direction (top down, z index up) and
    the units of dz.
    """
    table = tmp_path / "kappa.dat"
    _write_constant_kappa_table(table, 0.5)
    nz = 5
    mhd = MhdData(str(tmp_path), nx=1, ny=1, nz=nz)
    shape = (1, 1, nz)
    mhd.data = {"T": np.full(shape, 5800.0) * u.K, "P": np.full(shape, 1e5) * u.dyn / u.cm ** 2,
                "rho": np.full(shape, 1e-7) * u.g / u.cm ** 3}
    _quiet(mhd.load_opacity_table, str(table))
    logtau = _quiet(mhd.compute_optical_depth, dz=10 * u.km)
    tau = 10 ** logtau[0, 0]
    np.testing.assert_allclose(tau[:-1], 0.05 * np.arange(nz - 1, 0, -1), rtol=1e-9)
    assert logtau[0, 0, -1] == pytest.approx(-30.0)


def test_remap_recovers_a_function_linear_in_logtau():
    """Remapping a quantity that is linear in log tau returns it exactly at the new nodes.

    The remap is a per-column linear interpolation in log tau, so a linear profile must
    come back exactly, including at nodes outside the column's range (linear
    extrapolation, as happens for the deepest nodes of shallow columns).
    """
    mhd = MhdData(".", nx=2, ny=1, nz=6)
    mhd.logtau = np.stack([np.linspace(2.0, -3.0, 6), np.linspace(1.0, -4.0, 6)])[:, None, :]
    mhd.data = {"T": (6000.0 + 800.0 * mhd.logtau) * u.K}
    new = np.array([-3.5, -1.0, 0.0, 1.4])
    od = _quiet(mhd.remap_to_optical_depth, new, ["T"])
    np.testing.assert_allclose(od["T"].to_value(u.K), np.broadcast_to(6000.0 + 800.0 * new, (2, 1, 4)))


# --- StokesData --------------------------------------------------------------------------

def _gaussian_line(wl, center=6301.5, depth=0.6, sigma=0.03):
    """Absorption line on a continuum of 1."""
    return 1.0 - depth * np.exp(-((wl - center) ** 2) / (2 * sigma ** 2))


def test_continuum_normalization_modes():
    """Per-step mode makes the step's mean continuum exactly 1; fixed mode divides by the given level.

    All four Stokes parameters are divided by the same number, so V / I is unchanged.
    A non-positive fixed level is rejected.
    """
    s = StokesData(".", 110)
    s.data = {"I": np.array([[[3.0, 3.0, 1.0]], [[5.0, 5.0, 2.0]]]), "V": np.full((2, 1, 3), 0.8)}
    _, cont = _quiet(s.continuum_normalization, cont_indices=[0, 1])
    assert cont.mean() == pytest.approx(1.0)
    assert s.data["V"][0, 0, 0] == pytest.approx(0.2)

    s.data = {"I": np.full((1, 1, 3), 3.0), "V": np.full((1, 1, 3), 0.3)}
    _quiet(s.continuum_normalization, cont_indices=[0], fixed_ic=1.5)
    assert s.data["I"][0, 0, 0] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        _quiet(s.continuum_normalization, cont_indices=[0], fixed_ic=0.0)


@pytest.mark.skipif(not LSF_PATH.exists(), reason="Hinode LSF file absent")
@pytest.mark.data
def test_lsf_kernel_is_normalized_and_does_not_shift_lines():
    """The Hinode LSF kernel sums to 1, keeps the line area, and shifts a line by < 0.01 km/s.

    An off-centre kernel would shift every training line by the same amount, which the
    network would learn as a velocity offset. With the shipped file the kernel has 13
    points at 10 mA and moves a line by about 0.004 km/s. The equivalent width (A) must
    be conserved because the kernel is normalized.
    """
    s = StokesData(".", 110)
    kernel = _quiet(s.load_hinode_lsf, str(LSF_PATH))
    assert kernel.sum() == pytest.approx(1.0, abs=1e-12)
    line = _gaussian_line(s.wl)
    s.data = {"I": line[None, None, :].copy()}
    conv = _quiet(s.apply_spectral_convolution)["I"][0, 0]
    depth_in, depth_out = 1 - line, 1 - conv
    shift_aa = (depth_out * s.wl).sum() / depth_out.sum() - (depth_in * s.wl).sum() / depth_in.sum()
    assert abs(shift_aa / 6301.5 * C_KMS) < 0.01
    assert depth_out.sum() == pytest.approx(depth_in.sum(), rel=1e-9)
    assert depth_out.max() < depth_in.max()          # the line is broadened, so shallower


def test_resample_to_hinode_gives_112_points_on_the_fits_grid():
    """Resampling puts the profiles on the 112-point MODEST grid with cubic accuracy.

    A smooth line resampled from the 10 mA synthesis grid must match the line evaluated
    directly on the Hinode grid within 1e-4 (continuum units). Passing a grid shifted by
    the old hardcoded offset (0.0776 A, a spurious 3.7 km/s) must fail loudly.
    """
    from utils.hinode_wavelengths import hinode_wavelength_grid
    try:
        target = hinode_wavelength_grid()
    except FileNotFoundError:
        pytest.skip("MODEST FITS header absent")
    s = StokesData(".", 110)
    s.data = {"I": np.tile(_gaussian_line(s.wl), (2, 3, 1))}
    s.nx, s.ny, s.nwl = 2, 3, s.wl.size
    out = _quiet(s.resample_to_hinode)["I"]
    assert out.shape == (2, 3, 112)
    np.testing.assert_allclose(out[1, 2], _gaussian_line(target), atol=1e-4)
    np.testing.assert_array_equal(s.wl, target)

    s2 = StokesData(".", 110)
    s2.data = {"I": np.tile(_gaussian_line(s2.wl), (1, 1, 1))}
    s2.nx, s2.ny, s2.nwl = 1, 1, s2.wl.size
    with pytest.raises(AssertionError):
        _quiet(s2.resample_to_hinode, target - 0.0776)


def test_add_hinode_noise_has_requested_sigma():
    """Noise has sigma = noise_level times the continuum (here 1e-3) and zero mean, in every parameter."""
    s = StokesData(".", 110)
    s.data = {"I": np.ones((60, 60, 30)), "V": np.zeros((60, 60, 30))}
    s.nx, s.ny, s.nwl = 60, 60, 30
    np.random.seed(0)
    _quiet(s.add_hinode_noise, 1e-3, continuum_intensity=1.0)
    for key, base in (("I", 1.0), ("V", 0.0)):
        resid = s.data[key] - base
        assert resid.std() == pytest.approx(1e-3, rel=0.02)
        assert abs(resid.mean()) < 1e-5


# --- Pixel selection ---------------------------------------------------------------------

def test_otsu_threshold_separates_two_populations():
    """Otsu's threshold splits two well-separated populations (continuum 0.85 and 1.15).

    Any cut inside the empty gap between them is equally good, so the test checks the
    split itself: at least 99 % of the dark population falls below the threshold and all
    of the bright one above it.
    """
    rng = np.random.default_rng(0)
    dark = 0.85 + 0.02 * rng.standard_normal(500)
    bright = 1.15 + 0.02 * rng.standard_normal(300)
    t = _otsu_threshold(np.concatenate([dark, bright]))
    assert (dark <= t).mean() >= 0.99 and (bright > t).all()
    assert _otsu_threshold(np.full(10, 0.7)) == 0.7


def test_region_balancing_takes_equal_unique_pixels_from_each_region():
    """Each region contributes the smallest region's count, without repeats, and only its own pixels."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 4, size=(20, 20))
    names = ["granular_strong", "intergranular_strong", "granular_weak", "intergranular_weak"]
    masks = {n: labels == i for i, n in enumerate(names)}
    idx, info = build_balanced_region_indices(masks, rng)
    target = min(int(m.sum()) for m in masks.values())
    assert idx.size == 4 * target and np.unique(idx).size == idx.size
    for i, n in enumerate(names):
        assert int(np.sum(labels.ravel()[idx] == i)) == target


def test_bz_balancing_caps_oversampling_and_stays_within_candidates():
    """Rare strong-field pixels are oversampled at most ``max_oversample_factor`` times.

    With 990 weak and 10 strong pixels and linear bins, the target is the median bin
    count (500). The weak bin is subsampled to 500 distinct pixels, and the strong bin
    grows only to 10 x 10 = 100 draws, so a few pixels are not replicated hundreds of
    times. All indices must come from the candidate list.
    """
    rng = np.random.default_rng(0)
    bz = np.concatenate([np.full(990, 10.0), np.full(10, 1000.0)]) + rng.random(1000)
    bz = bz.reshape(10, 100, 1)
    candidates = np.arange(1000)
    idx, info = build_bz_strength_balanced_indices({"Bz": bz}, base_selected_indices=candidates, n_bins=2,
                                                   rng=rng, balance_cap=500.0, bin_scale="linear")
    strong = np.abs(bz.reshape(-1)[idx]) > 500
    assert info["target_per_bin"] == 500
    assert strong.sum() == 100 and np.unique(idx[strong]).size == 10
    assert (~strong).sum() == 500 and np.unique(idx[~strong]).size == 500
    assert np.isin(idx, candidates).all()


# --- MuramStepDataset --------------------------------------------------------------------

def test_dataset_targets_and_pixels_match_the_normalizer(tmp_path):
    """Each sample's targets are the normalized [T, Vz, Bz] of the pixel named by its (x, y).

    The target order (T, Vz, Bz, each by node) is what the model and the physics losses
    assume. The test compares every served sample with the normalizer's own output at its
    pixel, including repeated and reordered selections, and checks that out-of-range
    selections are refused. The input channels are covered by
    ``test_dataset_serves_four_stokes_channels_in_iquv_order``.
    """
    rng = np.random.default_rng(0)
    nx, ny, nwl, ntau = 3, 4, 6, 2
    stokes = {k: (1.0 if k == "I" else 0.0) + 1e-3 * rng.standard_normal((nx, ny, nwl)) for k in "IQUV"}
    mhd = {"T": 5000 + 100 * rng.random((nx, ny, ntau)), "Vz": rng.standard_normal((nx, ny, ntau)),
           "Bz": 100 * rng.standard_normal((nx, ny, ntau))}
    sn = StokesNormalizer()
    mn = MhdNormalizer(n_tau=ntau, raw_values_dir=str(tmp_path))
    sn.update(stokes)
    mn.update(mhd)
    _quiet(sn.finalize)
    _quiet(mn.finalize)
    nm = mn.transform(mhd)

    sel = np.array([11, 0, 5, 5])
    ds = MuramStepDataset(stokes, mhd, sn, mn, selected_flat_indices=sel)
    assert len(ds) == 4
    for k in range(len(ds)):
        _, y, xy = ds[k]
        ix, iy = xy.tolist()
        assert ix * ny + iy == sel[k]
        expected = np.concatenate([nm["T"][ix, iy], nm["Vz"][ix, iy], nm["Bz"][ix, iy]])
        np.testing.assert_allclose(y.numpy(), expected, rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        MuramStepDataset(stokes, mhd, sn, mn, selected_flat_indices=np.array([nx * ny]))


# --- Physical consistency of the training data -------------------------------------------

# The generator was fixed in reframe Phase 1a (v_LOS = -Vz, Bx and By passed to NICOLE) and
# every step was regenerated with it on 2026-11-27.

@pytest.mark.skipif(not (MURAM_DIR / "stokes_110_nicole_tau500.npy").exists(), reason="step 110 files absent")
@pytest.mark.data
def test_training_granules_are_blueshifted():
    """In the synthesized training profiles, bright granules must be blueshifted, as on the real Sun.

    Granules are hot upflows, so their Fe I 6302.5 A line core sits to the blue of the
    dark intergranular lanes. The test correlates the continuum with the line-core
    velocity (positive = redshift, from a parabola through the three lowest points) on
    every fourth pixel of step 110. On Hinode quiet Sun the correlation is -0.52.
    """
    s = np.load(MURAM_DIR / "stokes_110_nicole_tau500.npy", mmap_mode="r")
    wl = np.arange(6300.5, 6303.5, 0.01)[: s.shape[2]]
    I = np.asarray(s[::4, ::4, :, 0])
    ic = I[..., :4].mean(-1)
    win = (wl > 6302.2) & (wl < 6302.8)
    w, prof = wl[win], I[..., win]
    j = np.clip(prof.argmin(-1), 1, w.size - 2)
    ii = np.indices(j.shape)
    y0, y1, y2 = prof[ii[0], ii[1], j - 1], prof[ii[0], ii[1], j], prof[ii[0], ii[1], j + 1]
    core = w[j] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2) * (w[1] - w[0])
    v = (core - 6302.4936) / 6302.4936 * C_KMS
    assert np.corrcoef(ic.ravel(), v.ravel())[0, 1] < -0.2


# --- Targets for reframe Phases 1a and 1c: Q and U as inputs --------------------------------

IQUV_TARGET = ("Agreed in the reframe plan: Stokes Q and U become model inputs (Phase 1c), which "
               "needs the transverse field in the training synthesis (Phase 1a). Today only I and V are used.")


@pytest.mark.xfail(strict=True, reason=IQUV_TARGET)
def test_dataset_serves_four_stokes_channels_in_iquv_order(tmp_path):
    """Each sample's input is ``(4, n_wl)``: normalized I, Q, U, V in that order.

    The order is part of the model's contract: the convolution filters are trained per
    channel, so a model trained with one order cannot read another.
    """
    rng = np.random.default_rng(0)
    nx, ny, nwl, ntau = 2, 3, 5, 2
    stokes = {"I": 1 + 0.1 * rng.random((nx, ny, nwl)), "Q": 1e-3 * rng.standard_normal((nx, ny, nwl)),
              "U": 2e-3 * rng.standard_normal((nx, ny, nwl)), "V": 1e-3 * rng.standard_normal((nx, ny, nwl))}
    mhd = {"T": 5000 + rng.random((nx, ny, ntau)), "Vz": rng.random((nx, ny, ntau)),
           "Bz": 100 * rng.random((nx, ny, ntau))}
    sn, mn = StokesNormalizer(), MhdNormalizer(n_tau=ntau, raw_values_dir=str(tmp_path))
    sn.update(stokes)
    mn.update(mhd)
    _quiet(sn.finalize)
    _quiet(mn.finalize)
    ns = sn.transform(stokes)
    x, _, xy = MuramStepDataset(stokes, mhd, sn, mn)[4]
    ix, iy = xy.tolist()
    assert tuple(x.shape) == (4, nwl)
    np.testing.assert_allclose(x.numpy(), np.stack([ns[k][ix, iy] for k in ("I", "Q", "U", "V")]), rtol=1e-6)


@pytest.mark.skipif(not (MURAM_DIR / "stokes_110_nicole_tau500.npy").exists(), reason="step 110 files absent")
@pytest.mark.data
def test_training_profiles_carry_linear_polarization():
    """The synthesized training profiles have non-zero Q and U where the field is transverse.

    NICOLE only produces Q and U when the model includes Bx and By. Without them Q and U
    are about 1e-17 everywhere and adding them as inputs would teach the network nothing.
    Hinode's Q and U reach about 1e-2 of the continuum in strong transverse fields.
    """
    s = np.load(MURAM_DIR / "stokes_110_nicole_tau500.npy", mmap_mode="r")
    qu = np.abs(np.asarray(s[::4, ::4, :, 1:3]))
    assert qu.max() > 1e-3


def test_load_step_reads_the_horizontal_field(tmp_path):
    """With ``read_horizontal_field``, ``load_step`` returns Bx (result_7) and By (result_5) in G, laid out and scaled like Bz.

    Without the flag they are not read (they cost two extra cubes for callers that do not use
    them).
    """
    nx, ny, nz = 2, 3, 4
    x, y, z = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    code = (100 * x + 10 * y + z + 1).astype(np.float32)

    def to_file(arr_xyz):
        """Reorder an (nx, ny, nz) array to the (nx, nz, ny) file layout."""
        return np.ascontiguousarray(arr_xyz.transpose(0, 2, 1))

    ones = np.ones((nx, ny, nz), np.float32)
    np.stack([to_file(5000 * ones), to_file(1e5 * ones)]).tofile(tmp_path / "eos.110000")
    for name, arr in (("result_0", ones), ("result_2", ones), ("result_6", ones),
                      ("result_7", 2 * code), ("result_5", 3 * code)):
        to_file(arr).tofile(tmp_path / f"{name}.110000")
    data = _quiet(MhdData(str(tmp_path), nx=nx, ny=ny, nz=nz).load_step, 110, read_horizontal_field=True)
    assert "Bx" not in _quiet(MhdData(str(tmp_path), nx=nx, ny=ny, nz=nz).load_step, 110)
    np.testing.assert_allclose(data["Bx"].to_value(u.G), 2 * code * np.sqrt(4 * np.pi), rtol=1e-6)
    np.testing.assert_allclose(data["By"].to_value(u.G), 3 * code * np.sqrt(4 * np.pi), rtol=1e-6)


PSF_PATH = ROOT / "data" / "hinode-MODEST" / "PSFs" / "hinode_psf_bin.0.16.fits"


@pytest.mark.skipif(not PSF_PATH.exists(), reason="Hinode PSF file absent")
@pytest.mark.data
def test_hinode_psf_on_the_muram_grid_is_centred_normalized_and_keeps_its_wings():
    """The 0.16 arcsec Hinode PSF resampled to 25 km keeps sum 1, its centre and its encircled energy.

    25 km is 0.0345 arcsec at 1 AU, so the 9.8 arcsec file becomes 279 x 279 cells. The
    encircled energy inside 0.8 arcsec (0.858 in the file) must survive the resampling to
    0.01: the far wings are what lowers the granulation contrast to Hinode's level.
    """
    from utils.muram_data import KM_PER_ARCSEC, load_psf_on_muram_grid
    k = load_psf_on_muram_grid(PSF_PATH, 0.16, 25.0)
    assert k.shape == (279, 279) and k.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.unravel_index(k.argmax(), k.shape) == (139, 139)
    r = np.hypot(*np.mgrid[:279, :279] - 139) * 25.0 / KM_PER_ARCSEC
    assert k[r <= 0.8].sum() == pytest.approx(0.858, abs=0.01)


def test_spatial_psf_keeps_the_mean_and_lowers_the_contrast():
    """The PSF step blurs every wavelength map: the spatial mean is unchanged and the contrast drops.

    The convolution is normalized and periodic (the MURaM box is periodic), so the mean
    profile must be exactly conserved; on MURaM 198 the continuum contrast drops from 14.5 %
    to 8.4 % with the Hinode PSF (raw MODEST quiet Sun 6.9 %). Here a random map blurred by a Gaussian PSF must keep its mean to 1e-6
    and lose at least a third of its standard deviation.
    """
    rng = np.random.default_rng(0)
    s = StokesData(".", 110)
    s.data = {k: (1.0 if k == "I" else 0.0) + 0.1 * rng.standard_normal((32, 32, 3)) for k in "IQUV"}
    s.nx, s.ny, s.nwl = 32, 32, 3
    before = {k: v.copy() for k, v in s.data.items()}
    yy, xx = np.mgrid[-4:5, -4:5]
    psf = np.exp(-(xx ** 2 + yy ** 2) / (2 * 1.5 ** 2))
    _quiet(s.apply_spatial_psf, psf / psf.sum())
    for k in "IQUV":
        np.testing.assert_allclose(s.data[k].mean((0, 1)), before[k].mean((0, 1)), atol=1e-6)
        assert (s.data[k].std((0, 1)) < 0.67 * before[k].std((0, 1))).all()


@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1c): one helper, "
                   "utils.muram_data.stack_model_input, builds the model input everywhere.")
def test_stack_model_input_orders_channels_as_requested():
    """``stack_model_input`` flattens pixels and stacks the requested channels in the given order."""
    from utils.muram_data import stack_model_input
    norm = {k: np.full((2, 3, 5), float(i)) for i, k in enumerate("IQUV")}
    x = stack_model_input(norm, ("I", "Q", "U", "V"))
    assert x.shape == (6, 4, 5) and x.dtype == np.float32
    assert x[0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
