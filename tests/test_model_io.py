"""Round-trip tests for the NICOLE model/profile I/O in utils/model_prof_tools.py.

The writers are used to hand MURaM atmospheres and MUISCA predictions to NICOLE, so a
silent column or unit mistake would corrupt every synthesis. read_model / read_prof are
NICOLE's own readers, used here as the reference.
"""
import numpy as np
import pytest

from utils.model_prof_tools import (
    check_model,
    check_prof,
    read_model,
    read_prof,
    write_ascii_model,
    write_binary_model_cube,
    write_prof_cube,
)

LOGTAU = np.array([-2.0, -1.0, 0.0, 1.0])   # ascending, MUISCA convention (top -> bottom)
NZ = LOGTAU.size


def _var(data, ivar, nz=NZ):
    """Slice variable ``ivar`` (NICOLE order) out of a read_model record, deepest first."""
    return np.asarray(data[ivar * nz:(ivar + 1) * nz])


def test_binary_model_cube_round_trip(tmp_path):
    """Every pixel written by ``write_binary_model_cube`` reads back with NICOLE's ``read_model``.

    Checks log tau, T (K), v_LOS (written in cm/s, so km/s times 1e5), v_mic (cm/s) and
    B_long (G), in NICOLE's deepest-first order. NICOLE reads this file to synthesize the
    MURaM training profiles and the profiles of MUISCA's predictions. A column swap, a
    unit slip or a reversed depth order would corrupt every synthesis without an error.
    """
    rng = np.random.default_rng(0)
    nx, ny = 2, 3
    T = 5000 + 1000 * rng.random((nx, ny, NZ))
    v = rng.standard_normal((nx, ny, NZ))          # km/s
    b = 500 * rng.standard_normal((nx, ny, NZ))    # G
    path = tmp_path / "cube.model"
    write_binary_model_cube(path, LOGTAU, T, v, b, v_mic_cms=2.0e5)
    ftype, rnx, rny, rnz = check_model(str(path))
    assert (ftype, rnx, rny, rnz) == ("nicole2.3", nx, ny, NZ)
    order = np.argsort(LOGTAU)[::-1]                # file is deepest first
    for ix in range(nx):
        for iy in range(ny):
            d = read_model(str(path), ftype, nx, ny, NZ, ix, iy)
            np.testing.assert_allclose(_var(d, 1), LOGTAU[order])
            np.testing.assert_allclose(_var(d, 2), T[ix, iy, order])
            np.testing.assert_allclose(_var(d, 6), v[ix, iy, order] * 1e5)   # cm/s
            np.testing.assert_allclose(_var(d, 7), 2.0e5)
            np.testing.assert_allclose(_var(d, 8), b[ix, iy, order])          # B_long


def test_binary_model_cube_rejects_bad_shapes(tmp_path):
    """A depth axis that disagrees with log tau raises ValueError instead of writing a misaligned file."""
    with pytest.raises(ValueError):
        write_binary_model_cube(tmp_path / "x.model", LOGTAU, np.ones((2, 2, NZ)),
                                np.ones((2, 2, NZ)), np.ones((2, 2, NZ - 1)))


def test_ascii_model_round_trip(tmp_path):
    """The single-pixel ASCII model reads back, including the transverse field.

    Checks log tau, T, v_LOS, B_long and the transverse components b_x, b_y (G,
    variables 9 and 10), which NICOLE needs to produce non-zero Q and U. The tolerances
    match the number of decimals the file prints.
    """
    T = np.array([4500.0, 5000.0, 6000.0, 7000.0])
    v = np.array([0.5, -0.5, 1.0, 2.0])
    b = np.array([10.0, 20.0, 30.0, 40.0])
    path = tmp_path / "pixel.model"
    write_ascii_model(path, LOGTAU, T, v, b, b_x_G=5.0, b_y_G=-7.0)
    ftype, *_ = check_model(str(path))
    assert ftype.startswith("ascii")
    d = read_model(str(path), ftype, 1, 1, NZ, 0, 0)
    order = np.argsort(LOGTAU)[::-1]
    np.testing.assert_allclose(_var(d, 1), LOGTAU[order])
    np.testing.assert_allclose(_var(d, 2), T[order], atol=0.05)
    np.testing.assert_allclose(_var(d, 6), v[order] * 1e5, rtol=1e-3)
    np.testing.assert_allclose(_var(d, 8), b[order], atol=0.005)
    np.testing.assert_allclose(_var(d, 9), 5.0)
    np.testing.assert_allclose(_var(d, 10), -7.0)


def test_prof_cube_round_trip(tmp_path):
    """``write_prof_cube`` followed by ``read_prof`` gives back I, Q, U, V for every pixel and wavelength."""
    rng = np.random.default_rng(1)
    nx, ny, nlam = 2, 2, 5
    iquv = rng.standard_normal((nx, ny, 4, nlam))
    path = tmp_path / "obs.prof"
    write_prof_cube(path, iquv)
    ftype = "nicole2.3"   # check_prof cannot be used on binaries under Python 3, see below
    for ix in range(nx):
        for iy in range(ny):
            flat = np.asarray(read_prof(str(path), ftype, nx, ny, nlam, ix, iy))
            np.testing.assert_allclose(flat.reshape(nlam, 4).T, iquv[ix, iy])


@pytest.mark.xfail(raises=UnicodeDecodeError, strict=True,
                   reason="NICOLE's shipped check_prof opens files in text mode (Python 2 legacy) "
                          "and fails to decode binary profiles under Python 3; utils/synthesis.py "
                          "works around it by passing the known format to read_prof directly.")
def test_check_prof_on_binary_profiles(tmp_path):
    """Documents that NICOLE's shipped ``check_prof`` cannot read binary profiles in Python 3.

    It opens the file in text mode and raises UnicodeDecodeError. ``utils/synthesis.py``
    avoids it by passing the known format to ``read_prof``. The xfail is strict, so the test
    fails as a reminder if a newer NICOLE fixes the reader.
    """
    path = tmp_path / "obs.prof"
    write_prof_cube(path, np.random.default_rng(2).standard_normal((1, 1, 4, 5)))
    check_prof(str(path))


# --- Target for reframe Phase 1a: transverse field in the training synthesis -------------

@pytest.mark.xfail(strict=True, reason="Agreed in the reframe plan (Phase 1a): write_binary_model_cube takes "
                   "b_x_G and b_y_G so NICOLE can synthesize Q and U. Today the transverse slots are zero.")
def test_binary_model_cube_writes_the_transverse_field(tmp_path):
    """Bx and By (G) written to the binary cube read back in NICOLE's transverse-field slots 9 and 10.

    NICOLE's reader names slots 9 to 13 b_x, b_y, bl_x, bl_y, bl_z, while the writer's
    comment names them b_los_x, b_los_y, b_x, b_y, b_z. At disk centre both pairs describe
    the same field, so both are filled; the reader's slots 9 and 10 must hold Bx and By.
    """
    rng = np.random.default_rng(3)
    shape = (2, 2, NZ)
    T, v, b = 5000 + 100 * rng.random(shape), rng.standard_normal(shape), 100 * rng.standard_normal(shape)
    bx, by = 300 * rng.standard_normal(shape), 300 * rng.standard_normal(shape)
    path = tmp_path / "cube.model"
    write_binary_model_cube(path, LOGTAU, T, v, b, b_x_G=bx, b_y_G=by)
    order = np.argsort(LOGTAU)[::-1]
    d = read_model(str(path), "nicole2.3", 2, 2, NZ, 1, 0)
    np.testing.assert_allclose(_var(d, 9), bx[1, 0, order])
    np.testing.assert_allclose(_var(d, 10), by[1, 0, order])
