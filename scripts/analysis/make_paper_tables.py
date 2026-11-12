#!/usr/bin/env python3
"""Build the paper's MODEST and chi2 tables straight from the analysis outputs.

Writes one Markdown file with every table as a ready-to-paste LaTeX block, so no number in
the paper is typed by hand:

  - tab:modest_T, tab:modest_vlos, tab:modest_blos: correlation and RRMSE against the SPINOR 2D
    inversions, read from the `paired_pixel` rows of
    images/analysis/modest/cropped/<region>/<model>/metrics_summary.csv
    (written by tools/generate_analysis.sh --run modest --downsample-prediction-input 1
    --modest-pred-blos-invert-sign 1).
  - tab:chi2_I, tab:chi2_V: median chi2 of the NICOLE re-synthesis against the observed Stokes,
    over the pixels shared by all models, read from
    output/synthesis/<experiment>/modest/<region>/pixel_comparison/cross_model_chi2.json
    (written by tools/run_nicole_synthesis.sh).
  - tab:approx_accuracy: at the node where each physics term acts, the RRMSE and correlation
    against the MURaM step of the physics approximation alone, of the baseline and of the flavor
    trained with that term. The models come from
    images/analysis/muram/final/<step>/combined/metrics_vs_tau.csv and the approximations from the
    raw cache .muram_cache/step_<step>.h5, both written by
    tools/generate_analysis.sh --run muram --step-to-plot <step>.

The best value of each row (highest correlation, lowest RRMSE or chi2) is starred, using the
unrounded values.

Usage:
    python scripts/analysis/make_paper_tables.py [--output output/paper_tables/paper_tables.md]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.analysis import compute_regression_metrics  # noqa: E402

EXPERIMENT_ROOT = "experiment_110_to_130-step_size_10-normal"
EXPERIMENT_DIR = ROOT / "output" / "experiments" / EXPERIMENT_ROOT
MODEST_METRICS_DIR = ROOT / "images" / "analysis" / "modest" / "cropped"
MURAM_METRICS_DIR = ROOT / "images" / "analysis" / "muram" / "final"
MURAM_CACHE_DIR = ROOT / ".muram_cache"
SYNTHESIS_DIR = ROOT / "output" / "synthesis" / EXPERIMENT_ROOT / "modest"

# (flavor, label, approximation key in the cache, MURaM parameter, target-log-tau config key)
CONSTRAINTS = [
    ("wfa_only", "WFA", "blos", "Bz", "blos_target_logtau"),
    ("doppler_only", "Doppler", "vlos", "Vz", "vlos_target_logtau"),
    ("black_body_only", "Black Body", "temp", "T", "temp_target_logtau"),
]
APPROX_STEP = 130   # held out from training; the step hyperparameter decisions are made on

MODELS = ["no_physics", "wfa_only", "doppler_only", "black_body_only"]
MODEL_LABELS = ["Baseline", "WFA", "Doppler", "Black Body"]
REGIONS = ["quiet_sun", "negative_region", "plage", "sunspot"]
REGION_LABELS = {"quiet_sun": "Quiet Sun", "negative_region": "Negative region",
                 "plage": "Plage", "sunspot": "Sunspot"}
LOGTAUS = [-2.0, -0.8, 0.0]

# Scale of each chi2 table, matching the "(in units of 10^n)" of its caption
CHI2_SCALE = {"chi2_I": 1e5, "chi2_V": 1e3}

MODEST_TABLES = [
    ("T", "tab:modest_T",
     "Correlation and RRMSE between the MUISCA variants and the SPINOR 2D inversions of\n"
     "the MODEST observation, for the temperature, at the three optical depth nodes of the catalog\n"
     "and for each of the four selected regions. The best value of each row is shown with a star."),
    ("Vz", "tab:modest_vlos", "Same as Table~\\ref{tab:modest_T}, but for the line-of-sight velocity."),
    ("Bz", "tab:modest_blos", "Same as Table~\\ref{tab:modest_T}, but for the line-of-sight magnetic field."),
]

CHI2_TABLES = [
    ("chi2_I", "tab:chi2_I",
     "Median $\\chi^2_I$ (in units of $10^5$) between the NICOLE-resynthesized Stokes $I$\n"
     "and the observed MODEST Stokes $I$, per physics constraint and region, computed over the\n"
     "stratified $|B_{{\\rm LOS}}|$-binned pixel sample used for the synthesis consistency check\n"
     "({n_pixels} pixels per region). Lower is better; the best-performing constraint per region is\n"
     "shown with a star."),
    ("chi2_V", "tab:chi2_V",
     "Median $\\chi^2_V$ (in units of $10^3$) between the NICOLE-resynthesized Stokes $V$ and the "
     "observed MODEST Stokes $V$, per physics constraint and region, computed over the same pixel "
     "sample as table \\ref{{tab:chi2_I}}. Lower is better; the best-performing constraint per region "
     "is shown with a star."),
]


def fmt(value: float, best: bool) -> str:
    star = "^\\star" if best else ""
    return f"${value:.3f}{star}$"


def fmt_logtau(logtau: float) -> str:
    return f"$\\phantom{{-}}{logtau:.1f}$" if logtau >= 0 else f"${logtau:.1f}$"


def load_modest_metrics() -> dict:
    """{(region, model, param, logtau): (corr, rrmse)} from the paired-pixel rows."""
    metrics = {}
    for region in REGIONS:
        for model in MODELS:
            path = MODEST_METRICS_DIR / region / model / "metrics_summary.csv"
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["comparison"] != "paired_pixel":
                        continue
                    key = (region, model, row["param"], round(float(row["logtau"]), 1))
                    metrics[key] = (float(row["corr"]), float(row["rrmse"]))
    missing = [(r, m, p, t) for r in REGIONS for m in MODELS for p, _, _ in MODEST_TABLES
               for t in LOGTAUS if (r, m, p, t) not in metrics]
    if missing:
        raise ValueError(
            f"No paired-pixel metrics for {missing[:3]} ... ({len(missing)} in total). Was the "
            "MODEST analysis run with --downsample-prediction-input 1?"
        )
    return metrics


def modest_table(metrics: dict, param: str, label: str, caption: str) -> str:
    lines = [
        "\\begin{table*}[!htbp]", "\\centering", "\\footnotesize", "\\setlength{\\tabcolsep}{5pt}",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\begin{tabular}{llcccccccc}", "\\hline",
        " & & \\multicolumn{4}{c}{Correlation} & \\multicolumn{4}{c}{RRMSE} \\\\",
        "Region & $\\log \\tau$ & " + " & ".join(MODEL_LABELS * 2) + " \\\\", "\\hline",
    ]
    for region in REGIONS:
        for i, logtau in enumerate(LOGTAUS):
            corr = np.array([metrics[(region, m, param, logtau)][0] for m in MODELS])
            rrmse = np.array([metrics[(region, m, param, logtau)][1] for m in MODELS])
            cells = [fmt(v, k == int(np.argmax(corr))) for k, v in enumerate(corr)]
            cells += [fmt(v, k == int(np.argmin(rrmse))) for k, v in enumerate(rrmse)]
            name = REGION_LABELS[region] if i == 0 else ""
            lines.append(f"{name} & {fmt_logtau(logtau)} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def load_chi2() -> tuple[dict, dict]:
    """Median chi2 per (region, model, key) over the pixels shared by all models, and pixel counts."""
    medians, n_pixels = {}, {}
    for region in REGIONS:
        path = SYNTHESIS_DIR / region / "pixel_comparison" / "cross_model_chi2.json"
        with open(path, encoding="utf-8") as f:
            pixels = json.load(f)["pixels"]
        shared = [p for p in pixels.values() if all(m in p for m in MODELS)]
        n_pixels[region] = len(shared)
        for model in MODELS:
            for key in CHI2_SCALE:
                medians[(region, model, key)] = float(np.median([p[model][key] for p in shared]))
    return medians, n_pixels


def chi2_table(medians: dict, key: str, label: str, caption: str) -> str:
    lines = [
        "\\begin{table}[!htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}", "\\hline", "Region & " + " & ".join(MODEL_LABELS) + " \\\\", "\\hline",
    ]
    for region in REGIONS:
        values = np.array([medians[(region, m, key)] / CHI2_SCALE[key] for m in MODELS])
        cells = [fmt(v, k == int(np.argmin(values))) for k, v in enumerate(values)]
        lines.append(f"{REGION_LABELS[region]:<16} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def load_muram_metric(step: int, model: str, param: str, logtau: float) -> tuple[float, float]:
    """(corr, rrmse) of a model at one node of a MURaM step, from the vs-tau curves' CSV."""
    path = MURAM_METRICS_DIR / str(step) / "combined" / "metrics_vs_tau.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model and row["param"] == param
                    and np.isclose(float(row["logtau"]), logtau, atol=1e-4)):
                return float(row["corr"]), float(row["rrmse"])
    raise KeyError(f"No {param} at log tau = {logtau} in {path}")


def approx_accuracy_table(step: int) -> str:
    cache_path = MURAM_CACHE_DIR / f"step_{step:06d}.h5"
    rows = []
    with h5py.File(cache_path, "r") as h5:
        for flavor, label, approx_key, param, target_key in CONSTRAINTS:
            with open(EXPERIMENT_DIR / flavor / "experiment_config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            logtau = float(cfg["physics_config"][target_key])
            grid = np.asarray(cfg["data_config"]["logtau_values"])
            k = int(np.argmin(np.abs(grid - logtau)))
            approx = compute_regression_metrics(np.asarray(h5["mhd"][param][..., k]),
                                                np.asarray(h5["approximations"][approx_key][:]))
            base = load_muram_metric(step, "no_physics", param, logtau)
            flav = load_muram_metric(step, flavor, param, logtau)
            rows.append((label, logtau, (approx["rrmse"], base[1], flav[1]), (approx["corr"], base[0], flav[0])))

    caption = (
        f"RRMSE and correlation against MURaM step {step}, which is not used for training, at the\n"
        "optical depth where each physics term is applied, for the physics approximation on its own,\n"
        "the baseline, and the flavor trained with that term. A constraint pulls its node towards the\n"
        "approximation, so it can only improve that node when the approximation is more accurate\n"
        "than the baseline. The best value of each row is shown with a star."
    )
    lines = [
        "\\begin{table}[!htbp]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}",
        "\\label{tab:approx_accuracy}", "\\begin{tabular}{llcccccc}", "\\hline",
        " & & \\multicolumn{3}{c}{RRMSE} & \\multicolumn{3}{c}{Correlation} \\\\",
        "Constraint & $\\log \\tau$ & Approx. & Baseline & Flavor & Approx. & Baseline & Flavor \\\\", "\\hline",
    ]
    for label, logtau, rrmse, corr in rows:
        rrmse, corr = np.array(rrmse), np.array(corr)
        cells = [fmt(v, k == int(np.argmin(rrmse))) for k, v in enumerate(rrmse)]
        cells += [fmt(v, k == int(np.argmax(corr))) for k, v in enumerate(corr)]
        lines.append(f"{label} & {fmt_logtau(logtau)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "paper_tables" / "paper_tables.md")
    args = parser.parse_args()

    metrics = load_modest_metrics()
    medians, n_pixels = load_chi2()
    lo, hi = min(n_pixels.values()), max(n_pixels.values())
    n_text = f"{lo}" if lo == hi else f"{lo}--{hi}"

    parts = [
        "# MUISCA paper tables",
        "",
        f"Generated by `scripts/analysis/make_paper_tables.py` on {datetime.now():%Y-%m-%d %H:%M}.",
        "",
        f"- MODEST metrics: `{MODEST_METRICS_DIR.relative_to(ROOT)}/<region>/<model>/metrics_summary.csv` "
        "(paired-pixel rows)",
        f"- chi2: `{SYNTHESIS_DIR.relative_to(ROOT)}/<region>/pixel_comparison/cross_model_chi2.json` "
        f"(pixels shared by all models: "
        + ", ".join(f"{REGION_LABELS[r]} {n_pixels[r]}" for r in REGIONS) + ")",
        "",
    ]
    for param, label, caption in MODEST_TABLES:
        parts += [f"## {label}", "", "```latex", modest_table(metrics, param, label, caption), "```", ""]
    for key, label, caption in CHI2_TABLES:
        parts += [f"## {label}", "", "```latex",
                  chi2_table(medians, key, label, caption.format(n_pixels=n_text)), "```", ""]
    parts += ["## tab:approx_accuracy", "", "```latex", approx_accuracy_table(APPROX_STEP), "```", ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Tables written to {args.output}")


if __name__ == "__main__":
    main()
