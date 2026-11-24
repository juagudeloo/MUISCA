#!/usr/bin/env python3
"""Build the paper's MODEST and chi2 tables straight from the analysis outputs.

Writes one Markdown file with every table as a ready-to-paste LaTeX block, so no number in
the paper is typed by hand:

  - tab:modest_T, tab:modest_vlos, tab:modest_blos: correlation and RRMSE against the SPINOR 2D
    inversions, read from the `paired_pixel` rows of
    images/analysis/modest/cropped/<region>/<model>/metrics_summary.csv
    (written by tools/generate_analysis.sh --run modest --downsample-prediction-input 1
    --modest-pred-blos-invert-sign 1).
  - tab:chi2 (chi2_I and chi2_V side by side, full width): median chi2 of the NICOLE re-synthesis against the observed Stokes,
    over the pixels shared by all models, with its 95% bootstrap interval, read from
    output/synthesis/<experiment>/modest/<region>/pixel_comparison/cross_model_chi2.json
    (written by tools/run_nicole_synthesis.sh). No model is starred: which differences matter is
    argued in the text, and the Markdown under each table lists, per region, how often the lowest
    median stays below the runner-up's in paired resamples of the same pixels.
  - tab:approx_accuracy: at the node where each physics term acts, the RRMSE and correlation
    against the MURaM step of the physics approximation alone, of the baseline and of the flavor
    trained with that term. The models come from
    images/analysis/muram/final/<step>/combined/metrics_vs_tau.csv and the approximations from the
    raw cache .muram_cache/step_<step>.h5, both written by
    tools/generate_analysis.sh --run muram --step-to-plot <step>.

In the MODEST and approximation tables, the best value of each row (highest correlation, lowest
RRMSE) is starred, using the unrounded values.

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

# Bootstrap of the median chi2: resamples of each region's pixels, shared by all models so the
# comparison between models is paired. A fixed seed keeps the tables reproducible.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 0
CI_LEVEL = 0.95

MODEST_TABLES = [
    ("T", "tab:modest_T",
     "Correlation and RRMSE between the MUISCA variants and the SPINOR 2D inversions of\n"
     "the MODEST observation, for the temperature, at the three optical depth nodes of the catalog\n"
     "and for each of the four selected regions. The best value of each row is shown with a star."),
    ("Vz", "tab:modest_vlos", "Same as Table~\\ref{tab:modest_T}, but for the line-of-sight velocity."),
    ("Bz", "tab:modest_blos", "Same as Table~\\ref{tab:modest_T}, but for the line-of-sight magnetic field."),
]

CHI2_LABEL = "tab:chi2"
CHI2_CAPTION = (
    "Median $\\chi^2$ between the NICOLE-resynthesized and the observed MODEST Stokes $I$ and $V$, per\n"
    "physics constraint and region, computed over the\n"
    "stratified $|B_{{\\rm LOS}}|$-binned pixel sample used for the synthesis consistency check\n"
    "({n_pixels} pixels per region). $\\chi^2$ is dimensionless; the values shown are $\\chi^2_I$ divided by\n"
    "$10^5$ and $\\chi^2_V$ divided by $10^3$. Lower is better. Super- and subscripts give the 95\\%\n"
    "bootstrap interval of the median."
)


def fmt(value: float, best: bool) -> str:
    r"""A value as LaTeX with three decimals, starred when it is the best of its row.

    Args:
        value: The value.
        best: Add the star.

    Examples:
        >>> fmt(0.12345, True)
        '$0.123^\\star$'
    """
    star = "^\\star" if best else ""
    return f"${value:.3f}{star}$"


def fmt_logtau(logtau: float) -> str:
    r"""A node as LaTeX, padded so negative and positive nodes align.

    Args:
        logtau: The node.

    Examples:
        >>> fmt_logtau(-0.8), fmt_logtau(0.0)
        ('$-0.8$', '$\\phantom{-}0.0$')
    """
    return f"$\\phantom{{-}}{logtau:.1f}$" if logtau >= 0 else f"${logtau:.1f}$"


def load_modest_metrics() -> dict:
    """``{(region, model, param, logtau): (corr, rrmse)}`` from the paired-pixel rows of the MODEST CSVs.

    Returns:
        The metrics.

    Raises:
        ValueError: If a combination has no paired-pixel row (the MODEST analysis was not run
            with ``--downsample-prediction-input 1``).

    Examples:
        >>> m = load_modest_metrics()  # doctest: +SKIP
    """
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
    """The LaTeX table of correlation and RRMSE for one quantity: regions by rows, models by columns.

    The best value of each row is starred (highest correlation, lowest RRMSE), using the
    unrounded values.

    Args:
        metrics: Output of :func:`load_modest_metrics`.
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        label: LaTeX label.
        caption: Table caption.

    Returns:
        The LaTeX source.

    Examples:
        >>> print(modest_table(load_modest_metrics(), "T", "tab:modest_T", "..."))  # doctest: +SKIP
    """
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
    """chi^2 of every pixel shared by all models, as ``(n_pixels, n_models)`` arrays per ``(region, key)``.

    Returns:
        ``(samples, n_pixels_per_region)``.

    Examples:
        >>> samples, n = load_chi2()  # doctest: +SKIP
    """
    samples, n_pixels = {}, {}
    for region in REGIONS:
        path = SYNTHESIS_DIR / region / "pixel_comparison" / "cross_model_chi2.json"
        with open(path, encoding="utf-8") as f:
            pixels = json.load(f)["pixels"]
        shared = [p for p in pixels.values() if all(m in p for m in MODELS)]
        n_pixels[region] = len(shared)
        for key in CHI2_SCALE:
            samples[(region, key)] = np.array([[p[m][key] for m in MODELS] for p in shared])
    return samples, n_pixels


def chi2_stats(chi2: np.ndarray, rng: np.random.Generator) -> dict:
    """Median chi^2 per model, its bootstrap interval, and how often the best stays best.

    Pixels are resampled with replacement (the same resamples for every model, so the
    comparison is paired); ``p_rank`` is the fraction of resamples where the model with the
    lowest median stays below the runner-up.

    Args:
        chi2: ``(n_pixels, n_models)`` values.
        rng: Random generator (seeded for reproducible tables).

    Returns:
        Dict with ``median``, ``lo``, ``hi`` (per model), ``best``, ``runner_up`` and ``p_rank``.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> chi2 = np.column_stack([np.full(50, 1.0), np.full(50, 2.0)])
        >>> s = chi2_stats(chi2, rng)
        >>> s["median"].tolist(), s["best"], s["p_rank"]
        ([1.0, 2.0], 0, 1.0)
    """
    n = chi2.shape[0]
    resampled = np.median(chi2[rng.integers(0, n, size=(N_BOOTSTRAP, n))], axis=1)  # (N_BOOTSTRAP, n_models)
    tail = 100 * (1 - CI_LEVEL) / 2
    median = np.median(chi2, axis=0)
    best, runner_up = np.argsort(median)[:2]
    p_rank = float(np.mean(resampled[:, best] < resampled[:, runner_up]))
    return {
        "median": median,
        "lo": np.percentile(resampled, tail, axis=0),
        "hi": np.percentile(resampled, 100 - tail, axis=0),
        "best": int(best),
        "runner_up": int(runner_up),
        "p_rank": p_rank,
    }


def fmt_interval(median: float, lo: float, hi: float) -> str:
    """A median with its interval as LaTeX super- and subscripts.

    Args:
        median: The median.
        lo: Lower end of the interval.
        hi: Upper end of the interval.

    Examples:
        >>> fmt_interval(1.5, 1.25, 2.0)
        '$1.50^{+0.50}_{-0.25}$'
    """
    return f"${median:.2f}^{{+{hi - median:.2f}}}_{{-{median - lo:.2f}}}$"


def chi2_table(samples: dict, caption: str) -> tuple[str, dict]:
    """The full-width chi^2 table (I and V side by side) and, per key, a Markdown note on rank robustness.

    Args:
        samples: Output of :func:`load_chi2`.
        caption: Table caption.

    Returns:
        ``(latex_table, notes_by_key)``.

    Examples:
        >>> table, notes = chi2_table(samples, CHI2_CAPTION.format(n_pixels=40))  # doctest: +SKIP
    """
    stats = {}
    for key in CHI2_SCALE:
        rng = np.random.default_rng(BOOTSTRAP_SEED)   # same draws per key, independent of table layout
        stats[key] = {r: chi2_stats(samples[(r, key)] / CHI2_SCALE[key], rng) for r in REGIONS}

    n_models = len(MODELS)
    lines = [
        "\\begin{table*}[!htbp]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.6}",
        f"\\caption{{{caption}}}", f"\\label{{{CHI2_LABEL}}}",
        "\\begin{tabular}{l" + "c" * n_models + "c" + "c" * n_models + "}", "\\hline",
        f" & \\multicolumn{{{n_models}}}{{c}}{{$\\chi^2_I / 10^5$}} & & "
        f"\\multicolumn{{{n_models}}}{{c}}{{$\\chi^2_V / 10^3$}} \\\\",
        f"\\cline{{2-{n_models + 1}}} \\cline{{{n_models + 3}-{2 * n_models + 2}}}",
        "Region & " + " & ".join(MODEL_LABELS) + " & & " + " & ".join(MODEL_LABELS) + " \\\\", "\\hline",
    ]
    for r in REGIONS:
        blocks = [" & ".join(fmt_interval(stats[key][r]["median"][k], stats[key][r]["lo"][k],
                                          stats[key][r]["hi"][k]) for k in range(n_models))
                  for key in CHI2_SCALE]
        lines.append(f"{REGION_LABELS[r]:<16} & " + " & & ".join(blocks) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]

    notes = {}
    for key in CHI2_SCALE:
        note = ["| Region | Lowest median | Runner-up | P(rank holds) |", "|---|---|---|---|"]
        for r in REGIONS:
            st = stats[key][r]
            note.append(f"| {REGION_LABELS[r]} | {MODEL_LABELS[st['best']]} | "
                        f"{MODEL_LABELS[st['runner_up']]} | {st['p_rank']:.2f} |")
        notes[key] = "\n".join(note)
    return "\n".join(lines), notes


def load_muram_metric(step: int, model: str, param: str, logtau: float) -> tuple[float, float]:
    """``(corr, rrmse)`` of a model at one node of a MURaM step, from its ``metrics_vs_tau.csv``.

    Args:
        step: MURaM step.
        model: Flavor key.
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        logtau: The node.

    Returns:
        The two metrics.

    Raises:
        KeyError: If the CSV has no row for that model, quantity and node.

    Examples:
        >>> load_muram_metric(130, "wfa_only", "Bz", -0.8)  # doctest: +SKIP
    """
    path = MURAM_METRICS_DIR / str(step) / "combined" / "metrics_vs_tau.csv"
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (row["model"] == model and row["param"] == param
                    and np.isclose(float(row["logtau"]), logtau, atol=1e-4)):
                return float(row["corr"]), float(row["rrmse"])
    raise KeyError(f"No {param} at log tau = {logtau} in {path}")


def approx_accuracy_table(step: int) -> str:
    """Table of each approximation, the baseline and its flavor against MURaM at the term's node.

    Note:
        The approximations and the truth are read from ``.muram_cache/step_<step>.h5``
        without checking its configuration hash, and ``APPROX_STEP`` is the old validation
        step 130. Both are to be updated in reframe Phase 3 (hash check, step 150, five
        models).

    Args:
        step: MURaM step to evaluate on.

    Returns:
        The LaTeX source.

    Examples:
        >>> print(approx_accuracy_table(130))  # doctest: +SKIP
    """
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
    """Write every paper table to one Markdown file of LaTeX blocks.

    Examples:
        ::

            python scripts/analysis/make_paper_tables.py --output output/paper_tables/paper_tables.md
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path,
                        default=ROOT / "output" / "paper_tables" / "paper_tables.md")
    args = parser.parse_args()

    metrics = load_modest_metrics()
    chi2_samples, n_pixels = load_chi2()
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
    table, notes = chi2_table(chi2_samples, CHI2_CAPTION.format(n_pixels=n_text))
    parts += [f"## {CHI2_LABEL}", "", "```latex", table, "```", "",
              f"Paired bootstrap ({N_BOOTSTRAP} resamples): fraction of resamples in which the "
              "lowest median stays below the runner-up's.", ""]
    for key, note in notes.items():
        parts += [f"**{key}**", "", note, ""]
    parts += ["## tab:approx_accuracy", "", "```latex", approx_accuracy_table(APPROX_STEP), "```", ""]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts), encoding="utf-8")
    print(f"Tables written to {args.output}")


if __name__ == "__main__":
    main()
