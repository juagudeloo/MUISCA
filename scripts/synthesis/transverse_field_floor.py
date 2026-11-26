#!/usr/bin/env python3
"""How much chi^2 does omitting the transverse field cost? A model-independent floor for the paper.

MUISCA predicts T, v_LOS and B_LOS only, so the NICOLE re-synthesis of a prediction has no
transverse field, while the training (and observed) profiles do. In inclined fields the
re-synthesized I and V then differ from the real ones even for a perfect prediction. This
script measures that floor on MURaM's own atmospheres:

1. stratify the pixels of a step by |B| at log tau = -1 (the bins of
   ``utils.analysis.FIELD_STRENGTH_BIN_EDGES_G``) and draw a few per bin;
2. synthesize each true atmosphere with NICOLE twice on the Hinode grid, with its Bx and By
   and with Bx = By = 0 (v_LOS = -Vz and B_LOS = Bz in both);
3. compute chi^2_I and chi^2_V between the two (sigma = 1e-3 of the continuum, the value of
   the chi^2 bridge) and summarize them per |B| bin and per inclination to the line of sight.

Writes ``output/synthesis/transverse_field_floor/step-<step>/floor.json`` and ``floor.md``.

Usage:
    python scripts/synthesis/transverse_field_floor.py --step 198 --per-bin 30
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig  # noqa: E402
from scripts.synthesis.merge_tau500_stokes import count_bad_continua  # noqa: E402
from scripts.synthesis.tau500_multi_step_regen import NEW_LOGTAU_GRID, compute_logtau500  # noqa: E402
from utils.analysis import FIELD_STRENGTH_BIN_EDGES_G as FIELD_EDGES_G  # noqa: E402  (the per-bin metrics' |B| bins)
from utils.muram_data import MhdData  # noqa: E402
from utils.synthesis import NicoleRunner, SynthesisConfig, _rmtree_retry  # noqa: E402

INCLINATION_EDGES_DEG = (0.0, 30.0, 60.0, 90.0)
SIGMA = 1e-3
REFERENCE_LOGTAU = -1.0


def field_at_logtau(logtau: np.ndarray, field: dict[str, np.ndarray], target: float) -> dict[str, np.ndarray]:
    """Each field component at the layer whose log tau is closest to ``target``, per column.

    Nearest layer (10 km apart) rather than interpolation: it only serves to stratify pixels.

    Args:
        logtau: ``(nx, ny, nz)`` log tau of every cell.
        field: Components (e.g. ``Bx``, ``By``, ``Bz``) as ``(nx, ny, nz)`` arrays in G.
        target: The log tau of the reference layer.

    Returns:
        The components as ``(nx, ny)`` maps.

    Examples:
        >>> lt = np.array([[[1.0, 0.0, -1.0, -2.0]]])
        >>> field_at_logtau(lt, {"Bz": np.array([[[4.0, 3.0, 2.0, 1.0]]])}, -1.0)["Bz"].tolist()
        [[2.0]]
    """
    k = np.argmin(np.abs(logtau - target), axis=2)
    ix, iy = np.indices(k.shape)
    return {name: arr[ix, iy, k] for name, arr in field.items()}


def inclination_to_los_deg(bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> np.ndarray:
    """Angle between the field and the line of sight, folded to 0-90 degrees (polarity ignored).

    Args:
        bx: Field along x, G.
        by: Field along y, G.
        bz: Field along the line of sight, G.

    Returns:
        ``arccos(|Bz| / |B|)`` in degrees (0 for a vanishing field).

    Examples:
        >>> inclination_to_los_deg(np.array([0.0, 1.0]), np.array([0.0, 0.0]), np.array([-5.0, 1.0])).round(1).tolist()
        [0.0, 45.0]
    """
    b = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosg = np.where(b > 0, np.abs(bz) / b, 1.0)
    return np.degrees(np.arccos(np.clip(cosg, 0.0, 1.0)))


def select_pixels(abs_b: np.ndarray, per_bin: int, rng: np.random.Generator,
                  edges: tuple = FIELD_EDGES_G) -> np.ndarray:
    """Draw up to ``per_bin`` pixels from each |B| bin (without replacement).

    Args:
        abs_b: ``(nx, ny)`` |B| in G.
        per_bin: Pixels per bin (fewer if the bin is smaller).
        rng: Random generator.
        edges: Bin edges in G.

    Returns:
        ``(n, 2)`` pixel indices ``(ix, iy)``, bin by bin.

    Examples:
        >>> b = np.array([[50.0, 60.0], [300.0, 2000.0]])
        >>> select_pixels(b, 1, np.random.default_rng(0)).shape
        (3, 2)
    """
    chosen = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = np.argwhere((abs_b >= lo) & (abs_b < hi))
        if len(idx):
            take = rng.choice(len(idx), size=min(per_bin, len(idx)), replace=False)
            chosen.append(idx[take])
    return np.concatenate(chosen) if chosen else np.empty((0, 2), dtype=int)


def chi2_between(a: np.ndarray, b: np.ndarray, sigma: float = SIGMA) -> np.ndarray:
    """``sum(((a - b) / sigma)^2)`` over the last axis (wavelength), not divided by its length.

    Args:
        a: Profiles, ``(..., n_wl)``.
        b: Profiles, same shape.
        sigma: Noise in continuum units.

    Returns:
        One chi^2 per profile.

    Examples:
        >>> chi2_between(np.ones((1, 4)), np.ones((1, 4)) + 1e-3).round(6).tolist()
        [4.0]
    """
    return np.sum(((np.asarray(a) - np.asarray(b)) / sigma) ** 2, axis=-1)


def summarize(chi2: np.ndarray, values: np.ndarray, edges: tuple) -> list[dict]:
    """Count, median, 90th percentile and maximum of chi^2 within bins of ``values``.

    Args:
        chi2: One chi^2 per pixel.
        values: The binning quantity per pixel (|B| in G, or inclination in degrees).
        edges: Bin edges; the last bin includes its upper edge.

    Returns:
        One dict per non-empty bin with ``lo``, ``hi``, ``n``, ``median``, ``p90`` and ``max``.

    Examples:
        >>> rows = summarize(np.array([1.0, 3.0, 10.0]), np.array([10.0, 20.0, 80.0]), (0.0, 50.0, 90.0))
        >>> [(r["n"], r["median"]) for r in rows]
        [(2, 2.0), (1, 10.0)]
    """
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        m = (values >= lo) & ((values <= hi) if last else (values < hi))
        if m.any():
            c = chi2[m]
            rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "median": float(np.median(c)),
                         "p90": float(np.percentile(c, 90)), "max": float(c.max())})
    return rows


def _table(title: str, unit: str, rows_i: list[dict], rows_v: list[dict]) -> list[str]:
    """Markdown table of chi^2_I and chi^2_V medians and 90th percentiles per bin."""
    out = [f"### {title}", "", f"| Bin ({unit}) | n | chi2_I median | chi2_I p90 | chi2_V median | chi2_V p90 |",
           "|---|---|---|---|---|---|"]
    for ri, rv in zip(rows_i, rows_v):
        hi = "inf" if not np.isfinite(ri["hi"]) else f"{ri['hi']:g}"
        out.append(f"| {ri['lo']:g}-{hi} | {ri['n']} | {ri['median']:.3g} | {ri['p90']:.3g} | "
                   f"{rv['median']:.3g} | {rv['p90']:.3g} |")
    return out + [""]


def main() -> None:
    """Command-line entry point (see the module docstring).

    Examples:
        ::

            python scripts/synthesis/transverse_field_floor.py --step 198 --per-bin 30
    """
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--step", type=int, default=198)
    p.add_argument("--per-bin", type=int, default=30, help="pixels drawn per |B| bin")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=ROOT / "output" / "synthesis" / "transverse_field_floor")
    args = p.parse_args()

    cfg = TrainingConfig(checkpoint_dir=str(args.output_dir / "_ck"), log_dir=str(args.output_dir / "_lg"))
    out_dir = args.output_dir / f"step-{args.step}"
    out_dir.mkdir(parents=True, exist_ok=True)

    mhd = MhdData(cfg.data_path / "muram-simulation", nx=cfg.nx, ny=cfg.ny, nz=cfg.nz)
    with contextlib.redirect_stdout(io.StringIO()):
        mhd.load_step(args.step, z_max=cfg.z_max, read_horizontal_field=True)
    logtau = compute_logtau500(mhd, dz_km=cfg.dz_km)
    q = {k: mhd.data[k].to_value(u.G) for k in ("Bx", "By", "Bz")}
    ref = field_at_logtau(logtau, q, REFERENCE_LOGTAU)
    abs_b = np.sqrt(ref["Bx"] ** 2 + ref["By"] ** 2 + ref["Bz"] ** 2)
    pixels = select_pixels(abs_b, args.per_bin, np.random.default_rng(args.seed))
    print(f"Step {args.step}: {len(pixels)} pixels in {len(FIELD_EDGES_G) - 1} |B| bins")

    T = mhd.data["T"].to_value(u.K)
    v_los = -mhd.data["Vz"].to_value(u.km / u.s)  # km/s, positive = redshift (NICOLE's convention)
    G = NEW_LOGTAU_GRID
    cols = {k: np.empty((1, len(pixels), G.size)) for k in ("T", "v", "Bz", "Bx", "By")}
    sources = (("T", T), ("v", v_los), ("Bz", q["Bz"]), ("Bx", q["Bx"]), ("By", q["By"]))
    for n, (ix, iy) in enumerate(pixels):
        lt = logtau[ix, iy]
        for key, arr in sources:
            cols[key][0, n] = interp1d(lt, arr[ix, iy], bounds_error=False, fill_value="extrapolate")(G)

    runner = NicoleRunner(SynthesisConfig(source="muram", experiment_root="transverse_field_floor",
                                          model_type="truth", muram_step=args.step, output_root=args.output_dir))
    synth = {}
    for name, kw in (("with", dict(b_x_G=cols["Bx"], b_y_G=cols["By"])), ("without", {})):
        wd = out_dir / f"_nicole_{name}"
        synth[name] = runner.run_cube(wd, G, cols["T"], cols["v"], cols["Bz"], runner.cfg.n_wl, **kw)[0]
        _rmtree_retry(wd)
        n_bad = count_bad_continua(np.transpose(synth[name], (0, 2, 1)))
        if n_bad:
            sys.exit(f"{n_bad} pixels with an unphysical continuum in the '{name}' synthesis; refusing to report.")

    chi2_i = chi2_between(synth["with"][:, 0], synth["without"][:, 0])
    chi2_v = chi2_between(synth["with"][:, 3], synth["without"][:, 3])
    b_pix = abs_b[pixels[:, 0], pixels[:, 1]]
    incl = inclination_to_los_deg(*(ref[k][pixels[:, 0], pixels[:, 1]] for k in ("Bx", "By", "Bz")))

    result = {
        "step": args.step, "n_pixels": int(len(pixels)), "sigma": SIGMA, "reference_logtau": REFERENCE_LOGTAU,
        "n_wl": int(runner.cfg.n_wl), "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
        "by_field": {"chi2_I": summarize(chi2_i, b_pix, FIELD_EDGES_G), "chi2_V": summarize(chi2_v, b_pix, FIELD_EDGES_G)},
        "by_inclination": {"chi2_I": summarize(chi2_i, incl, INCLINATION_EDGES_DEG),
                           "chi2_V": summarize(chi2_v, incl, INCLINATION_EDGES_DEG)},
        "pixels": [{"ix": int(ix), "iy": int(iy), "abs_b_G": float(b), "inclination_deg": float(g),
                    "chi2_I": float(ci), "chi2_V": float(cv)}
                   for (ix, iy), b, g, ci, cv in zip(pixels, b_pix, incl, chi2_i, chi2_v)],
    }
    (out_dir / "floor.json").write_text(json.dumps(result, indent=2))
    md = [f"# Transverse-field chi2 floor, MURaM step {args.step}", "",
          f"True atmospheres synthesized with and without Bx, By ({len(pixels)} pixels, sigma = {SIGMA}, "
          f"{runner.cfg.n_wl} Hinode wavelengths, |B| and inclination at log tau = {REFERENCE_LOGTAU}).", ""]
    md += _table("By |B|", "G", result["by_field"]["chi2_I"], result["by_field"]["chi2_V"])
    md += _table("By inclination to the line of sight", "deg", result["by_inclination"]["chi2_I"],
                 result["by_inclination"]["chi2_V"])
    (out_dir / "floor.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"Written to {out_dir}")


if __name__ == "__main__":
    main()
