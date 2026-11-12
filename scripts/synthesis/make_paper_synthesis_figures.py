#!/usr/bin/env python3
"""Build the paper's NICOLE re-synthesis figures: one Stokes I and one Stokes V panel per pixel.

For the six pixels in CASES (chosen to span |B_LOS| from the quiet Sun to the umbra, see
output/synthesis/<experiment>/modest/synthesis_for_paper/SELECTION_NOTES.md), each panel overlays
the observed profile and the NICOLE re-synthesis from every model, with each model's chi2 in a box.
Reads the per-model predictions.h5, syntheses.h5 and comparison/chi2.json written by
tools/run_nicole_synthesis.sh, and writes <basename>_I.png / <basename>_V.png to
output/synthesis/<experiment>/modest/synthesis_for_paper/. Nothing is written into the paper.

Usage:
    python scripts/synthesis/make_paper_synthesis_figures.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.analysis import display_name

EXPERIMENT_ROOT = "experiment_110_to_130-step_size_10-normal"
SYNTH_BASE = ROOT / "output" / "synthesis" / EXPERIMENT_ROOT / "modest"
MODEL_TYPES = ["no_physics", "wfa_only", "doppler_only", "black_body_only"]
MODEL_COLORS = {"no_physics": "C1", "wfa_only": "C2", "doppler_only": "C3", "black_body_only": "C4"}

# (region_label, ix, iy, display_region_name, output_basename)
CASES = [
    ("quiet_sun", 17, 179, "Quiet Sun", "quiet_sun_BLOS5.6G_pix017_179"),
    ("plage", 17, 97, "Plage", "plage_BLOS43G_pix017_097"),
    ("negative_region", 2, 198, "Negative region", "negative_region_BLOS74G_pix002_198"),
    ("plage", 12, 1, "Plage", "plage_BLOS686G_pix012_001"),
    ("plage", 65, 301, "Plage", "plage_BLOS1670G_pix065_301"),
    ("sunspot", 106, 277, "Sunspot", "sunspot_BLOS2370G_pix106_277"),
]

OUT_DIR = SYNTH_BASE / "synthesis_for_paper"


def load_case(region_label: str, ix: int, iy: int):
    pred_path = SYNTH_BASE / MODEL_TYPES[0] / region_label / "predictions.h5"
    with h5py.File(pred_path, "r") as h5:
        pred_pixels = h5["pixels"][...]
        stokes_obs = h5["stokes_obs"][...]
        wl = h5["wavelengths"][...]
    pred_index = {(int(a), int(b)): i for i, (a, b) in enumerate(pred_pixels.tolist())}
    i_obs = stokes_obs[pred_index[(ix, iy)], 0]
    v_obs = stokes_obs[pred_index[(ix, iy)], 1]

    sel_path = SYNTH_BASE / MODEL_TYPES[0] / region_label / "pixel_selection" / "selected_pixels.json"
    abs_b_los = None
    if sel_path.exists():
        sel = json.load(open(sel_path))
        for p in sel["pixels"]:
            if p["ix"] == ix and p["iy"] == iy:
                abs_b_los = p.get("abs_bz_gauss")
                break

    synth_by_model = {}
    for mt in MODEL_TYPES:
        synth_path = SYNTH_BASE / mt / region_label / "syntheses.h5"
        with h5py.File(synth_path, "r") as h5:
            synth_pixels = h5["pixels"][...]
            stokes_synth = h5["stokes_synth"][...]
        synth_index = {(int(a), int(b)): i for i, (a, b) in enumerate(synth_pixels.tolist())}
        row = stokes_synth[synth_index[(ix, iy)]]
        synth_by_model[mt] = (row[0], row[3])

    chi2_by_model = {}
    for mt in MODEL_TYPES:
        chi2_path = SYNTH_BASE / mt / region_label / "comparison" / "chi2.json"
        d = json.load(open(chi2_path))
        chi2_by_model[mt] = d.get(f"{ix},{iy}")

    return wl, i_obs, v_obs, synth_by_model, abs_b_los, chi2_by_model


def _format_sci(value: float) -> str:
    if value is None or not np.isfinite(value) or value == 0:
        return "n/a"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10 ** exp)
    return f"${mant:.2f}\\times10^{{{exp}}}$"


def make_panel(wl, obs, synth_by_model, stokes_idx, ylabel, region_name, b_label,
               chi2_by_model, chi2_key, out_path):
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    ax.plot(wl, obs, color="k", lw=1.8, label="Observed")
    for mt in MODEL_TYPES:
        y = synth_by_model[mt][stokes_idx]
        ax.plot(wl, y, color=MODEL_COLORS[mt], ls="--", lw=1.5, label=display_name(mt))

    ax.set_xlabel("Wavelength [Å]", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.25)
    ax.set_title(f"{region_name} -- |B$_{{\\rm LOS}}$| = {b_label}", fontsize=14, fontweight="bold")
    # Below the axes, not "best" -- with 5 lines "best" can land on top of the data itself
    # (e.g. a line-core dip that happens to sit in the legend's preferred corner).
    ax.legend(fontsize=10, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)

    # Carve out guaranteed empty headroom above the data for the chi2 box -- the continuum
    # level varies a lot between cases (weak-field continuum sits close to the autoscaled top),
    # so a fixed-position box can otherwise land right on top of real curves in some cases.
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.34 * (yhi - ylo))

    chi2_lines = [
        f"{display_name(mt)}: {_format_sci((chi2_by_model[mt] or {}).get(chi2_key))}"
        for mt in MODEL_TYPES
    ]
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)
    ax.text(0.97, 0.97, r"$\chi^2$" + "\n" + "\n".join(chi2_lines), transform=ax.transAxes,
            fontsize=9, ha="right", va="top", bbox=props)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.family"] = "serif"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for region_label, ix, iy, region_name, basename in CASES:
        wl, i_obs, v_obs, synth_by_model, abs_b_los, chi2_by_model = load_case(region_label, ix, iy)
        b_label = f"{abs_b_los:.3g} G" if abs_b_los is not None else "?"
        make_panel(wl, i_obs, synth_by_model, 0, r"$I/I_c$", region_name, b_label,
                   chi2_by_model, "chi2_I", OUT_DIR / f"{basename}_I.png")
        make_panel(wl, v_obs, synth_by_model, 1, r"$V/I_c$", region_name, b_label,
                   chi2_by_model, "chi2_V", OUT_DIR / f"{basename}_V.png")
        print(f"Saved {basename}_I.png / _V.png")


if __name__ == "__main__":
    main()
