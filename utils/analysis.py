"""Post-training analysis: load trained flavors, predict on MURaM or MODEST, compute metrics and plot.

* Metric helpers (:func:`compute_regression_metrics`,
  :func:`compute_distribution_similarity_metrics`, :func:`compute_metrics_by_field_strength`,
  :func:`compute_metric_vs_tau`). RRMSE is always ``RMSE / mean(|truth|)``, dimensionless.
* :class:`AnalysisModelPipeline`: finds the flavors of an experiment, loads their
  checkpoints and predicts in physical units.
* :class:`MuramDiagnosticPlots` and :class:`ModestDiagnosticPlots`: per-model maps,
  scatter plots and ``metrics_summary.csv`` (the source of the paper's MODEST tables).
* Multi-model figures (:func:`plot_combined_images_grid`,
  :func:`plot_combined_jointplot_grid`, :func:`plot_metric_vs_tau`).
* Labels: display names, LaTeX symbols and units for every quantity.

Examples:
    The scripts ``scripts/analysis/muram_analysis.py`` and ``modest_analysis.py`` drive
    this module; in short::

        pipe = AnalysisModelPipeline(device, output_dir, experiment_root=root)
        configs, models, n_tau = pipe.prepare_models(["no_physics", "wfa_only"])
        pred = pipe.predict_and_denormalize(models["wfa_only"], stokes_input, mhd_norm, nx, ny)
"""

from pathlib import Path
import json
import csv
import torch
import torch.nn.functional as F
from typing import Callable, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from scripts.base_training import LEGACY_DEGRADATION, TrainingConfig
from models.pinn_mscnn_model import PhysicsInformedMSCNN
from utils.modest_data import transform_modest_stokes_profiles

try:
    from torchinfo import summary as torch_summary
except Exception:
    torch_summary = None


# Edges (G) for reporting accuracy against the true |B_LOS| of each pixel. Chosen around the
# training data's support: MURaM steps 110-130 hold ~25k distinct pixels above 200 G and ~7k
# above 352 G, but only ~150 above 954 G and none at all above 1500 G, while the sunspot
# crops reach 3.7 kG. Splitting the metrics here separates "the model was trained for this"
# from "the model is extrapolating".
FIELD_STRENGTH_BIN_EDGES_G = (0.0, 100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, float("inf"))


# Human-readable labels for plot titles/legends only. Every other use of an experiment key
# (output directories, filenames, CSV "model" columns, model_configs dict keys) stays as the
# raw key -- those are load-bearing for path resolution and downstream filtering elsewhere in
# the pipeline, and must not be renamed.
MODEL_DISPLAY_NAMES = {
    "no_physics": "Baseline",
    "wfa_only": "WFA",
    "doppler_only": "Doppler",
    "black_body_only": "Black Body",
}

# Fixed left-to-right/top-to-bottom order for multi-model comparison figures, independent of
# whatever order --model-types was given in on the command line.
MODEL_CANONICAL_ORDER = ("no_physics", "wfa_only", "doppler_only", "black_body_only")


def display_name(model_type: str) -> str:
    """Plot label of a flavor's key ("wfa_only" gives "WFA"); unknown keys are returned unchanged.

    Only for titles and legends: folder names, file names and CSV columns keep the raw key.

    Args:
        model_type: Flavor key.

    Returns:
        The display name.

    Examples:
        >>> display_name("no_physics"), display_name("wfa-lambda-10")
        ('Baseline', 'wfa-lambda-10')
    """
    return MODEL_DISPLAY_NAMES.get(model_type, model_type)


# LaTeX-formatted symbol for each internal MHD parameter key, for plot titles only. MURaM is
# synthesised at disc centre, where the vertical components are the line-of-sight ones. The
# field keeps $B_z$ (positive up, i.e. toward the observer, the B_LOS sign). The velocity under
# the "Vz" key is the training target, v_LOS = -Vz (positive = redshift, away from the
# observer), so it is labelled v_LOS: labelling it V_z would show granules (upflows) as negative
# under an upward-positive symbol. MODEST figures compare against SPINOR's line-of-sight
# quantities and say so for both, which is what `los=True` selects.
PARAM_LATEX = {"T": r"$T$", "Vz": r"$v_{\mathrm{LOS}}$", "Bz": r"$B_z$"}
PARAM_LATEX_LOS = {"T": r"$T$", "Vz": r"$v_{\mathrm{LOS}}$", "Bz": r"$B_{\mathrm{LOS}}$"}


def param_latex(param: str, los: bool = False) -> str:
    r"""LaTeX symbol of a quantity: ``$B_z$``, or ``$B_{\mathrm{LOS}}$`` with ``los=True``.

    MURaM figures use ``$B_z$`` for the field (at disk centre it is B_LOS, same sign) and
    ``$v_{\mathrm{LOS}}$`` for the velocity, which is the target v_LOS = -Vz (positive =
    redshift). MODEST figures compare with SPINOR's line-of-sight quantities and use
    ``los=True``.

    Args:
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        los: Use the line-of-sight symbols.

    Returns:
        The LaTeX string.

    Examples:
        >>> param_latex("Bz"), param_latex("Bz", los=True)
        ('$B_z$', '$B_{\\mathrm{LOS}}$')
    """
    table = PARAM_LATEX_LOS if los else PARAM_LATEX
    return table.get(param, f"${param}$")


# Physical unit of each MHD parameter. Every quantity plotted here must carry its unit;
# quantities that are genuinely dimensionless (the Stokes profiles, normalized to the
# continuum) are labelled as the explicit ratio instead, so the reader sees why.
PARAM_UNITS = {"T": "K", "Vz": r"$\mathrm{km\,s^{-1}}$", "Bz": "G"}


def param_unit(param: str) -> str:
    """Unit label of a quantity: ``"K"``, km/s in LaTeX, or ``"G"`` (empty for unknown keys).

    Args:
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.

    Examples:
        >>> param_unit("T"), param_unit("Bz")
        ('K', 'G')
    """
    return PARAM_UNITS.get(param, "")


def param_axis_label(param: str, prefix: str = "", los: bool = False) -> str:
    """Axis or colourbar label with its unit, e.g. ``"$B_z$ [G]"`` or ``"Ground truth [G]"``.

    Args:
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        prefix: Text replacing the symbol (e.g. ``"Pred - GT"``).
        los: Use the line-of-sight symbols.

    Returns:
        The label.

    Examples:
        >>> param_axis_label("Bz"), param_axis_label("T", "Ground truth")
        ('$B_z$ [G]', 'Ground truth [K]')
    """
    head = prefix if prefix else param_latex(param, los=los)
    unit = param_unit(param)
    return f"{head} [{unit}]" if unit else head


# Quantile clip for every image colour scale: 2nd--98th percentile. Kept as one constant so the
# per-model panels and the combined grids never drift apart.
IMAGE_COLOR_QUANTILE = 0.02

# Helioprojective geometry of the MODEST scan, the same values used for the observation
# figures of the paper: FOV centre (Solar-X, Solar-Y) in arcsec and the Hinode/SP fast-map
# pixel scale. Crops are located with these so a map can be placed inside the full scene.
MODEST_FOV_CENTER_ARCSEC = (16.0, -126.0)
MODEST_PIXEL_SCALE_ARCSEC = 0.32


def modest_crop_extent(
    crop_bounds: tuple[int, int, int, int],
    scene_shape: tuple[int, int],
) -> tuple[tuple[float, float, float, float], tuple[str, str]]:
    """Helioprojective extent (arcsec) of a MODEST crop given in native pixels ``(y0, y1, x0, x1)``.

    The extent is for the untransposed orientation (horizontal axis Solar-X, vertical
    Solar-Y), the one the full-scene and regions-overview figures use, so maps must be drawn
    with ``transpose=False``. It is computed from the native 0.32 arcsec bounds, so it does
    not depend on whether the predictions are on the upsampled or the native grid; pixel
    edges (not centres) are mapped, hence the half-pixel offset.

    Args:
        crop_bounds: ``(y0, y1, x0, x1)`` on the native grid.
        scene_shape: ``(ny, nx)`` of the whole scene.

    Returns:
        ``((x_left, x_right, y_bottom, y_top), ("Solar-X [arcsec]", "Solar-Y [arcsec]"))``.

    Examples:
        >>> ext, labels = modest_crop_extent((0, 390, 0, 858), (390, 858))
        >>> round(ext[1] - ext[0], 2), round(ext[3] - ext[2], 2), labels[0]
        (274.56, 124.8, 'Solar-X [arcsec]')
    """
    y0, y1, x0, x1 = (float(v) for v in crop_bounds)
    ny, nx = (float(v) for v in scene_shape)
    x_center, y_center = MODEST_FOV_CENTER_ARCSEC
    scale = MODEST_PIXEL_SCALE_ARCSEC

    def to_x(px: float) -> float:
        """Solar-X (arcsec) of a native column coordinate."""
        return x_center + (px - nx / 2.0) * scale

    def to_y(py: float) -> float:
        """Solar-Y (arcsec) of a native row coordinate."""
        return y_center + (py - ny / 2.0) * scale

    # imshow's extent maps array EDGES, not pixel centres -- hence the half-pixel offset.
    extent = (to_x(x0 - 0.5), to_x(x1 - 0.5), to_y(y0 - 0.5), to_y(y1 - 0.5))
    return extent, ("Solar-X [arcsec]", "Solar-Y [arcsec]")


def combined_plot_title(param: str, context: str, tau_val: float, los: bool = False) -> str:
    r"""Title of the multi-model figures: ``"<symbol> - <context> | $\log \tau = <tau>$"``.

    Args:
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        context: e.g. ``"Simulation step 198"``.
        tau_val: The node.
        los: Use the line-of-sight symbols.

    Examples:
        >>> combined_plot_title("T", "Simulation step 198", -1.0)
        '$T$ - Simulation step 198 | $\\log \\tau = -1.00$'
    """
    return f"{param_latex(param, los=los)} - {context} | " + r"$\log \tau = " + f"{tau_val:.2f}" + r"$"


def combined_plot_title_no_tau(param: str, context: str, los: bool = False) -> str:
    r"""Same title without a node, for figures whose x axis is log tau.

    Args:
        param: ``"T"``, ``"Vz"`` or ``"Bz"``.
        context: e.g. ``"Simulation step 198"``.
        los: Use the line-of-sight symbols.

    Examples:
        >>> combined_plot_title_no_tau("Vz", "MODEST - Sunspot", los=True)
        '$v_{\\mathrm{LOS}}$ - MODEST - Sunspot'
    """
    return f"{param_latex(param, los=los)} - {context}"


def ordered_models(models: dict) -> list[str]:
    """Keys of ``models`` in the canonical order, unknown keys (e.g. lambda variants) appended in their order.

    Args:
        models: Dict keyed by flavor.

    Returns:
        The ordered keys.

    Examples:
        >>> ordered_models({"wfa-lambda-1": 0, "wfa_only": 0, "no_physics": 0})
        ['no_physics', 'wfa_only', 'wfa-lambda-1']
    """
    ordered = [m for m in MODEL_CANONICAL_ORDER if m in models]
    ordered += [m for m in models if m not in MODEL_CANONICAL_ORDER]
    return ordered


def compute_metrics_by_field_strength(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    field_strength: np.ndarray,
    edges: tuple = FIELD_STRENGTH_BIN_EDGES_G,
) -> list[dict]:
    """Regression metrics within bins of the reference |B_LOS| (G) of each pixel.

    A field of view is dominated by weak pixels, so a good overall correlation can hide a
    badly wrong strong-field regime. Binning on the reference field (SPINOR's or MURaM's),
    not on the prediction, keeps the split independent of the model. The default edges are
    0, 100, 250, 500, 750, 1000, 1500 G and above.

    Args:
        y_true: Reference values.
        y_pred: Predictions, same shape.
        field_strength: Reference B_LOS per pixel in G (its absolute value is used).
        edges: Bin edges in G.

    Returns:
        One dict per bin with ``field_bin``, ``bin_lo_G``, ``bin_hi_G``, ``n_points`` and,
        for non-empty bins, the metrics of :func:`compute_regression_metrics`.

    Examples:
        >>> rows = compute_metrics_by_field_strength(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 0.0]),
        ...                                          np.array([50.0, 60.0, 2000.0]))
        >>> [(r["field_bin"], r["n_points"]) for r in rows if r["n_points"]]
        [('0-100G', 2), ('>1500G', 1)]
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    strength = np.abs(np.asarray(field_strength).ravel())

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (strength >= lo) & (strength < hi) & np.isfinite(y_true) & np.isfinite(y_pred)
        n = int(mask.sum())
        label = f"{lo:.0f}-{hi:.0f}G" if np.isfinite(hi) else f">{lo:.0f}G"
        if n == 0:
            rows.append({"field_bin": label, "bin_lo_G": float(lo), "bin_hi_G": float(hi), "n_points": 0})
            continue
        metrics = compute_regression_metrics(y_true[mask], y_pred[mask])
        metrics.update({"field_bin": label, "bin_lo_G": float(lo), "bin_hi_G": float(hi)})
        rows.append(metrics)
    return rows


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Paired metrics of predictions against a reference, over the finite pairs.

    ``rmse``, ``mae`` and ``bias`` (mean of pred - truth) are in the data's units;
    ``rrmse`` = rmse / mean(|truth|) and ``nmae`` = mae / mean(|truth|) are dimensionless;
    ``corr`` is Pearson's r (NaN if either side is constant); ``r2`` = 1 - SS_res / SS_tot.

    Args:
        y_true: Reference values (any shape).
        y_pred: Predictions, same number of elements.

    Returns:
        Dict with ``n_points``, ``rmse``, ``rrmse``, ``mae``, ``nmae``, ``bias``, ``corr``,
        ``r2`` (NaN when no pair is finite).

    Examples:
        >>> m = compute_regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 4.0]))
        >>> round(m["rmse"], 4), round(m["rrmse"], 4), round(m["bias"], 4)
        (0.5774, 0.2887, 0.3333)
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[m]
    y_pred = y_pred[m]

    if y_true.size == 0:
        return {
            "n_points": 0,
            "rmse": np.nan,
            "rrmse": np.nan,
            "mae": np.nan,
            "nmae": np.nan,
            "bias": np.nan,
            "corr": np.nan,
            "r2": np.nan,
        }

    diff = y_pred - y_true
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    denom = float(np.mean(np.abs(y_true)) + 1e-10)

    if y_true.size > 1 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = np.nan

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else np.nan

    return {
        "n_points": int(y_true.size),
        "rmse": rmse,
        "rrmse": float(rmse / denom),
        "mae": mae,
        "nmae": float(mae / denom),
        "bias": bias,
        "corr": corr,
        "r2": r2,
    }


def format_metric(value: float, decimals: int = 3) -> str:
    """A metric as text with ``decimals`` decimals, or ``"nan"``.

    Args:
        value: The metric.
        decimals: Number of decimals.

    Examples:
        >>> format_metric(0.12345), format_metric(float("nan"))
        ('0.123', 'nan')
    """
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{decimals}f}"


def compute_distribution_similarity_metrics(x: np.ndarray, y: np.ndarray, bins: int = 128) -> dict[str, float]:
    """Unpaired comparison of two value distributions (used when the grids differ).

    * ``ks``: two-sample Kolmogorov-Smirnov statistic, 0 to 1.
    * ``w1_quantile``: 1-Wasserstein distance from up to 20 000 matched quantiles, in the
      data's units.
    * ``jsd``: Jensen-Shannon divergence of ``bins``-bin histograms on a shared range
      (natural log, 0 to ln 2).
    * ``overlap``: sum of the bin-wise minimum of the two normalized histograms, 0 to 1.

    Args:
        x: First sample (the reference).
        y: Second sample.
        bins: Number of histogram bins.

    Returns:
        Dict with ``n_true``, ``n_pred`` and the four metrics (NaN if a sample is empty).

    Examples:
        >>> rng = np.random.default_rng(0); a = rng.standard_normal(5000)
        >>> same = compute_distribution_similarity_metrics(a, a)
        >>> same["ks"], round(same["overlap"], 6)
        (0.0, 1.0)
    """
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    if x.size == 0 or y.size == 0:
        return {
            "n_true": int(x.size),
            "n_pred": int(y.size),
            "ks": np.nan,
            "w1_quantile": np.nan,
            "jsd": np.nan,
            "overlap": np.nan,
        }

    # KS statistic (two-sample) using ECDFs
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    grid = np.sort(np.concatenate([x_sorted, y_sorted]))
    cdf_x = np.searchsorted(x_sorted, grid, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, grid, side="right") / y_sorted.size
    ks = float(np.max(np.abs(cdf_x - cdf_y)))

    # 1-Wasserstein approximation via quantile functions
    n_q = min(20000, x.size, y.size)
    q = np.linspace(0.0, 1.0, n_q, endpoint=False) + 0.5 / n_q
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    w1_quantile = float(np.mean(np.abs(xq - yq)))

    # Histogram-based Jensen-Shannon divergence + overlap coefficient
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        jsd = np.nan
        overlap = np.nan
    else:
        hist_x, edges = np.histogram(x, bins=bins, range=(lo, hi), density=False)
        hist_y, _ = np.histogram(y, bins=edges, density=False)
        px = hist_x.astype(np.float64)
        py = hist_y.astype(np.float64)
        px /= max(px.sum(), 1.0)
        py /= max(py.sum(), 1.0)

        eps = 1e-12
        m = 0.5 * (px + py)
        kl_pm = np.sum(px * (np.log(px + eps) - np.log(m + eps)))
        kl_qm = np.sum(py * (np.log(py + eps) - np.log(m + eps)))
        jsd = float(0.5 * (kl_pm + kl_qm))
        overlap = float(np.sum(np.minimum(px, py)))

    return {
        "n_true": int(x.size),
        "n_pred": int(y.size),
        "ks": ks,
        "w1_quantile": w1_quantile,
        "jsd": jsd,
        "overlap": overlap,
    }


class AnalysisModelPipeline:
    """Find an experiment's flavors, load their checkpoints and predict in physical units.

    Note:
        :meth:`load_model` and the model summary assume 2 input channels and the default
        architecture; a 4-channel checkpoint will not load (strict xfail in
        ``tests/test_analysis.py``; reframe Phase 1c).
    """

    def __init__(
        self,
        device,
        output_dir: Path | None = None,
        experiments_base_dir: Path | str | None = None,
        experiment_root: str = "experiment_80_to_113",
    ):
        """Set the device and folders.

        Args:
            device: Torch device for inference.
            output_dir: Folder for the analysis outputs (also the runtime config's ``log_dir``).
            experiments_base_dir: Folder holding the experiments (``output/experiments`` by default).
            experiment_root: Experiment folder name.
        """
        self.device = device
        self.output_dir = output_dir
        self.experiments_base_dir = Path(experiments_base_dir) if experiments_base_dir is not None else Path(
            "/scratchsan/observatorio/juagudeloo/MUISCA/output/experiments/"
        )
        self.experiment_root = str(experiment_root)

    def get_model_configs(self) -> dict:
        """One entry per flavor folder of the experiment that has a ``final_model.pth``.

        Each entry holds the checkpoint, configuration and results paths, the experiment key,
        the lambdas (from ``experiment_config.json``), a label and a colour.

        Returns:
            Dict keyed by experiment key.

        Raises:
            FileNotFoundError: If the experiment folder or every checkpoint is missing.

        Examples:
            >>> AnalysisModelPipeline("cpu", Path("out"), experiment_root=root).get_model_configs()  # doctest: +SKIP
        """
        experiment_dir = self.experiments_base_dir / self.experiment_root
        results_path = experiment_dir / "experiment_results.json"
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment root not found: {experiment_dir}")

        model_configs = {}
        for subdir in sorted([p for p in experiment_dir.iterdir() if p.is_dir()]):
            model_path = subdir / "final_model.pth"
            config_path = subdir / "experiment_config.json"
            if not model_path.exists():
                continue

            experiment_key = subdir.name
            lambda_wfa = 0.0
            lambda_doppler = 0.0
            lambda_temp = 0.0

            if config_path.exists():
                try:
                    with open(config_path, "r") as f:
                        raw_cfg = json.load(f)
                    experiment_key = raw_cfg.get("experiment_name", experiment_key)
                    phys_cfg = raw_cfg.get("physics_config", {})
                    lambda_wfa = float(phys_cfg.get("lambda_wfa", 0.0))
                    lambda_doppler = float(phys_cfg.get("lambda_doppler", 0.0))
                    lambda_temp = float(phys_cfg.get("lambda_temp", 0.0))
                except Exception:
                    pass

            color = "gray"
            if experiment_key.startswith("no_physics"):
                color = "blue"
            elif experiment_key.startswith("wfa_only"):
                color = "orange"
            elif experiment_key.startswith("doppler_only"):
                color = "green"
            elif experiment_key.startswith("black_body_only"):
                color = "red"
            elif experiment_key.startswith("all_physics_terms"):
                color = "purple"

            key_name = experiment_key if experiment_key not in model_configs else f"{experiment_key}__{subdir.name}"
            model_configs[key_name] = {
                "path": model_path,
                "config_path": config_path,
                "results_path": results_path,
                "experiment_key": experiment_key,
                "output_subdir": subdir.name,
                "use_physics": None,
                "lambda_wfa": lambda_wfa,
                "lambda_doppler": lambda_doppler,
                "lambda_temp": lambda_temp,
                "label": f"{experiment_key} ({self.experiment_root})",
                "color": color,
            }

        if not model_configs:
            raise FileNotFoundError(
                f"No models with final_model.pth found under {experiment_dir}"
            )

        return model_configs

    @staticmethod
    def select_model_configs(model_configs: dict, selected_types: list[str]) -> dict:
        """Keep the flavors named in ``selected_types`` (``"all"`` or empty keeps all).

        Short names are accepted: ``wfa``, ``doppler``, ``black-body``/``black_body``,
        ``all_physics`` and the folder forms ``wfa-lambda-<x>``. A name also selects its lambda
        variants (``wfa_only`` selects ``wfa_only-lambda-10``).

        Args:
            model_configs: Output of :meth:`get_model_configs`.
            selected_types: Requested flavors.

        Returns:
            The filtered dict.

        Raises:
            ValueError: If nothing matches.

        Examples:
            >>> cfgs = {"wfa_only": {"experiment_key": "wfa_only"}, "no_physics": {"experiment_key": "no_physics"}}
            >>> list(AnalysisModelPipeline.select_model_configs(cfgs, ["wfa"]))
            ['wfa_only']
        """
        if not selected_types or "all" in selected_types:
            return model_configs

        def _normalize_selected_type(value: str) -> str:
            """Map a short or folder-style flavor name to its experiment key."""
            if value.startswith("wfa-lambda-"):
                return value.replace("wfa-lambda-", "wfa_only-lambda-", 1)
            if value.startswith("doppler-lambda-"):
                return value.replace("doppler-lambda-", "doppler_only-lambda-", 1)
            if value.startswith("black-body-lambda-"):
                return value.replace("black-body-lambda-", "black_body_only-lambda-", 1)
            if value.startswith("black_body-lambda-"):
                return value.replace("black_body-lambda-", "black_body_only-lambda-", 1)
            if value == "wfa":
                return "wfa_only"
            if value == "doppler":
                return "doppler_only"
            if value in {"black-body", "black_body"}:
                return "black_body_only"
            if value == "all_physics":
                return "all_physics_terms"
            return value

        selected_set = {_normalize_selected_type(s) for s in selected_types}
        filtered = {
            name: cfg for name, cfg in model_configs.items()
            if (
                cfg.get("experiment_key") in selected_set
                or any(
                    cfg.get("experiment_key", "").startswith(f"{sel}-lambda-")
                    for sel in selected_set
                )
            )
        }
        if not filtered:
            available = sorted({cfg.get("experiment_key") for cfg in model_configs.values()})
            raise ValueError(
                f"No models matched --model-types={selected_types}. "
                f"Available types: {available}"
            )
        return filtered

    def build_runtime_training_config(self, model_cfg: dict) -> TrainingConfig:
        """A :class:`TrainingConfig` with the flavor's physics settings, for re-running its data pipeline.

        Missing entries fall back to the old defaults (``tau_averaged`` B_LOS, v_LOS at -1.0,
        T at 0.0), and a run without ``spatial_psf`` or ``hinode_noise`` (or without a saved
        configuration) is replayed without PSF and noise, as it was trained.

        Args:
            model_cfg: One entry of :meth:`get_model_configs`.

        Returns:
            The configuration, with ``log_dir`` set to the pipeline's output folder.

        Examples:
            >>> cfg = pipe.build_runtime_training_config(configs["wfa_only"])  # doctest: +SKIP
        """
        cfg_kwargs = {
            "device": str(self.device),
            "batch_size": 512,
            "enable_epoch_plots": True,
            **LEGACY_DEGRADATION,  # without a saved config, assume a pre-Phase-1b run
        }

        cfg_path = model_cfg.get("config_path")
        if cfg_path is not None and Path(cfg_path).exists():
            with open(cfg_path, "r") as f:
                raw = json.load(f)

            train_cfg = raw.get("training_config", {})
            phys_cfg = raw.get("physics_config", {})
            data_cfg = raw.get("data_config", {})

            cfg_kwargs.update({
                "data_path": raw.get("data_path", TrainingConfig().data_path),
                "learning_rate": train_cfg.get("learning_rate", 1e-3),
                "batch_size": int(train_cfg.get("batch_size", 512)),
                "lambda_wfa": float(phys_cfg.get("lambda_wfa", model_cfg["lambda_wfa"])),
                "lambda_doppler": float(phys_cfg.get("lambda_doppler", model_cfg["lambda_doppler"])),
                "lambda_temp": float(phys_cfg.get("lambda_temp", model_cfg["lambda_temp"])),
                "blos_physics_mode": phys_cfg.get("blos_physics_mode", "tau_averaged"),
                "blos_target_logtau": phys_cfg.get("blos_target_logtau", None),
                "vlos_physics_mode": phys_cfg.get("vlos_physics_mode", "single_height"),
                "vlos_target_logtau": phys_cfg.get("vlos_target_logtau", -1.0),
                "temp_physics_mode": phys_cfg.get("temp_physics_mode", "single_height"),
                "temp_target_logtau": phys_cfg.get("temp_target_logtau", 0.0),
                "temp_continuum_indices": phys_cfg.get("temp_continuum_indices", None),
                "temp_continuum_wavelength": phys_cfg.get("temp_continuum_wavelength", None),
                "logtau_values": data_cfg.get("logtau_values", None),
                # Runs saved before Phase 1b were trained without PSF and noise.
                "spatial_psf": bool(data_cfg.get("spatial_psf", LEGACY_DEGRADATION["spatial_psf"])),
                "hinode_noise": float(data_cfg.get("hinode_noise", LEGACY_DEGRADATION["hinode_noise"])),
            })
            for key in ("psf_path", "psf_sampling_arcsec", "pixel_km"):
                if key in data_cfg:
                    cfg_kwargs[key] = data_cfg[key]

        cfg = TrainingConfig(**cfg_kwargs)
        cfg.log_dir = self.output_dir
        cfg.log_dir.mkdir(parents=True, exist_ok=True)
        return cfg

    def load_model(self, config: dict) -> tuple[PhysicsInformedMSCNN, int]:
        """Load one flavor's checkpoint (evaluation mode).

        The number of nodes is read from the output layer; the rest of the architecture is fixed
        (scales 1, 2, 3; 2 input channels; 16 filters; kernel 5; pool 2; 4 dense layers).

        Args:
            config: One entry of :meth:`get_model_configs`.

        Returns:
            ``(model, n_tau)``.

        Raises:
            ValueError: If the output size is not a multiple of 3.

        Examples:
            >>> model, n_tau = pipe.load_model(configs["wfa_only"])  # doctest: +SKIP
        """
        checkpoint = torch.load(config["path"], map_location=self.device)
        output_features = self._infer_output_features_from_checkpoint(checkpoint)
        if output_features % 3 != 0:
            raise ValueError(f"Invalid output_features={output_features}; expected multiple of 3")
        n_tau = output_features // 3

        model = PhysicsInformedMSCNN(
            scales=[1, 2, 3],
            in_channels=2,
            c1_filters=16,
            c2_filters=32,
            kernel_size=5,
            pool_size=2,
            n_linear_layers=4,
            output_features=output_features,
            input_length=112,
            lambda_wfa=config["lambda_wfa"],
            lambda_doppler=config["lambda_doppler"],
            lambda_temp=config["lambda_temp"],
        ).to(self.device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, n_tau

    @staticmethod
    def _infer_output_features_from_checkpoint(checkpoint: dict) -> int:
        """Output size of a checkpoint, from the output layer's bias."""
        state = checkpoint["model_state_dict"]
        key = "linear_block.output_layer.bias"
        if key not in state:
            raise KeyError(f"Missing '{key}' in checkpoint state_dict")
        return int(state[key].shape[0])

    def load_all_models(self, model_configs: dict) -> tuple[dict, int]:
        """Load every flavor and check that they share one number of nodes.

        Args:
            model_configs: Entries of :meth:`get_model_configs`.

        Returns:
            ``(models, n_tau)``.

        Raises:
            ValueError: If two checkpoints have different numbers of nodes.

        Examples:
            >>> models, n_tau = pipe.load_all_models(configs)  # doctest: +SKIP
        """
        models = {}
        n_tau_ref = None
        for name, config in model_configs.items():
            print(f"Loading {config['label']}...")
            model, n_tau = self.load_model(config)
            if n_tau_ref is None:
                n_tau_ref = n_tau
            elif n_tau != n_tau_ref:
                raise ValueError(f"Inconsistent n_tau across models: {name} has {n_tau}, expected {n_tau_ref}")
            models[name] = model
            print(f"  ✓ Model loaded successfully (n_tau={n_tau})")
            self._print_model_summary(model, config["label"], self.device)
        print(f"\n✓ All {len(models)} models loaded\n")
        return models, int(n_tau_ref)

    def prepare_models(self, selected_types: list[str]) -> tuple[dict, dict, int]:
        """Find, select and load the requested flavors.

        Args:
            selected_types: Requested flavors (see :meth:`select_model_configs`).

        Returns:
            ``(model_configs, models, n_tau)``.

        Examples:
            >>> configs, models, n_tau = pipe.prepare_models(["no_physics", "wfa_only"])  # doctest: +SKIP
        """
        model_configs = self.get_model_configs()
        model_configs = self.select_model_configs(model_configs, selected_types)
        models, n_tau = self.load_all_models(model_configs)
        return model_configs, models, n_tau

    @staticmethod
    def _print_model_summary(model: PhysicsInformedMSCNN, model_label: str, device) -> None:
        """Print a torchinfo summary for a (1, 2, 112) input, or the module and parameter counts."""
        print(f"\n--- Torch summary: {model_label} ---")
        if torch_summary is not None:
            try:
                torch_summary(
                    model,
                    input_size=(1, 2, 112),
                    device=str(device),
                    col_names=("input_size", "output_size", "num_params"),
                    depth=3,
                    verbose=1,
                )
                return
            except Exception as e:
                print(f"torchinfo summary failed: {e}")

        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(model)
        print(f"Total params: {n_params:,}")
        print(f"Trainable params: {n_trainable:,}")

    def get_model_logtau_values(self, model_cfg: dict) -> list[float] | None:
        """The flavor's log tau nodes, from ``experiment_results.json`` or else ``experiment_config.json``.

        Args:
            model_cfg: One entry of :meth:`get_model_configs`.

        Returns:
            The nodes, or ``None`` if neither file has them.

        Examples:
            >>> pipe.get_model_logtau_values(configs["wfa_only"])[:2]  # doctest: +SKIP
            [-3.0, -2.9]
        """
        exp_key = model_cfg.get("experiment_key")
        results_path = model_cfg.get("results_path")

        if exp_key and results_path and Path(results_path).exists():
            try:
                with open(results_path, "r") as f:
                    raw = json.load(f)
                vals = raw.get(exp_key, {}).get("config", {}).get("logtau_values", None)
                if vals is not None:
                    return [float(v) for v in vals]
            except Exception:
                pass

        cfg_path = model_cfg.get("config_path")
        if cfg_path and Path(cfg_path).exists():
            try:
                with open(cfg_path, "r") as f:
                    raw = json.load(f)
                vals = raw.get("data_config", {}).get("logtau_values", None)
                if vals is not None:
                    return [float(v) for v in vals]
            except Exception:
                pass

        return None

    def predict_and_denormalize(
        self,
        model: PhysicsInformedMSCNN,
        stokes_input: np.ndarray,
        mhd_normalizer,
        pred_nx: int,
        pred_ny: int,
        batch_size: int = 4096,
    ) -> dict[str, np.ndarray]:
        """Predict on a flat input and return ``(nx, ny, n_tau)`` cubes in K, km/s and G.

        Denormalization uses :meth:`MhdNormalizer.denormalize` (including its physical clip of
        Bz and its Vz mean offset, see the normalizer's strict xfail). No resizing is done.

        Args:
            model: The model.
            stokes_input: ``(nx * ny, n_channels, n_wl)`` normalized input.
            mhd_normalizer: Fitted MHD normalizer.
            pred_nx: Size of the first map axis.
            pred_ny: Size of the second map axis.
            batch_size: Inference batch size.

        Returns:
            ``{"T", "Vz", "Bz"}`` cubes.

        Raises:
            ValueError: If the output is not ``(N, 3 n_tau)``.

        Examples:
            >>> pred = pipe.predict_and_denormalize(model, x, mhd_norm, 390, 858)  # doctest: +SKIP
        """
        model.eval()
        all_pred = []

        with torch.no_grad():
            for i in range(0, stokes_input.shape[0], batch_size):
                x = torch.from_numpy(stokes_input[i:i + batch_size]).float().to(self.device)
                y = model(x).detach().cpu().numpy()
                all_pred.append(y)

        pred = np.concatenate(all_pred, axis=0)  # (N, 3*n_tau) block order [T|Vz|Bz]
        if pred.ndim != 2 or pred.shape[1] % 3 != 0:
            raise ValueError(f"Expected prediction shape (N, 3*n_tau), got {pred.shape}")
        n_tau_local = pred.shape[1] // 3

        T_denorm = mhd_normalizer.denormalize(pred[:, :n_tau_local], param="T").reshape(pred_nx, pred_ny, n_tau_local)
        Vz_denorm = mhd_normalizer.denormalize(pred[:, n_tau_local:2 * n_tau_local], param="Vz").reshape(pred_nx, pred_ny, n_tau_local)
        Bz_denorm = mhd_normalizer.denormalize(pred[:, 2 * n_tau_local:3 * n_tau_local], param="Bz").reshape(pred_nx, pred_ny, n_tau_local)
        return {"T": T_denorm, "Vz": Vz_denorm, "Bz": Bz_denorm}

    @staticmethod
    def common_tau_matches(
        modest_tau: list[float],
        pred_tau: list[float],
        atol: float = 1e-6,
    ) -> list[tuple[float, int, int]]:
        """Nodes present in both grids, as ``(tau, index_in_modest, index_in_prediction)`` (match to ``atol``).

        Args:
            modest_tau: Reference nodes (e.g. SPINOR's -2.0, -0.8, 0.0).
            pred_tau: The model's nodes.
            atol: Matching tolerance.

        Returns:
            The matches, in the reference order.

        Examples:
            >>> AnalysisModelPipeline.common_tau_matches([-2.0, -0.8, 0.0], [-2.0, -1.0, -0.8, 0.0])
            [(-2.0, 0, 0), (-0.8, 1, 2), (0.0, 2, 3)]
        """
        matches: list[tuple[float, int, int]] = []
        for i_m, mt in enumerate(modest_tau):
            for i_p, pt in enumerate(pred_tau):
                if np.isclose(float(mt), float(pt), atol=atol, rtol=0.0):
                    matches.append((float(mt), i_m, i_p))
                    break
        return matches

class MuramDiagnosticPlots:
    """Per-model maps, scatter plots and metrics of predictions against MURaM at chosen nodes.

    Outputs go to ``<output_dir>/final/<step>/<model_name>/``: for each node of
    ``config.epoch_plot_ods`` (default -1.0, -0.8, 0.0) and quantity, an image panel and a
    joint plot, then ``metrics_summary.csv`` and the mean Stokes profiles.
    """

    def __init__(
        self,
        config: TrainingConfig,
        model_name: str,
        label: str | None = None,
        step: int | None = None,
        output_dir: str | Path | None = Path('./images'),
        stokes_normalizer=None,
    ):
        """Set up the output folder and plotting options.

        Args:
            config: Training configuration (nodes, quantities, scatter sample size).
            model_name: Flavor key.
            label: Snapshot label for titles (default ``step_<step>``).
            step: MURaM step (output subfolder).
            output_dir: Base output folder.
            stokes_normalizer: Fitted Stokes normalizer, to undo the input normalization in the
                profile plot.
        """
        self.config = config
        self.model_name = model_name
        self.label = label if label is not None else (f"step_{step}" if step is not None else "snapshot")

        self.logtau = config.get_logtau_values()
        self.ods = config.epoch_plot_ods if config.epoch_plot_ods is not None else [-1.0, -0.8, 0.0]
        self.params = config.epoch_plot_params if config.epoch_plot_params is not None else ["T", "Vz", "Bz"]
        self.n_sample = int(config.epoch_plot_scatter_samples)

        self.param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
        self.error_cmap = "RdBu_r"
        self.stokes_normalizer = stokes_normalizer
        
        base_out_dir = Path(output_dir)
        step_folder = str(step) if step is not None else "snapshot"
        self.out_dir = base_out_dir / "final" / step_folder / model_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_rows: list[dict[str, float | str | int]] = []

    def plot_stokes_mean_std(
        self,
        stokes_input: np.ndarray,
        wavelengths: np.ndarray | None = None,
    ) -> None:
        """Plot the mean and standard deviation of the model's I and V input, in continuum units.

        Silently does nothing unless the input has exactly 2 channels (I, V).

        Args:
            stokes_input: ``(n, 2, n_wl)`` normalized input.
            wavelengths: The wavelength axis in A (required, from the data's metadata).

        Raises:
            ValueError: If ``wavelengths`` is missing or has the wrong length.

        Examples:
            >>> plots.plot_stokes_mean_std(x, wavelengths=wl)  # doctest: +SKIP
        """
        arr = np.asarray(stokes_input, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2:
            return

        if wavelengths is None:
            raise ValueError(
                "wavelengths parameter is required and must come from MODEST metadata. "
                "Do not use hardcoded Hinode wavelength defaults."
            )
        
        wl = np.asarray(wavelengths, dtype=np.float64)
        if wl.ndim != 1 or wl.size != arr.shape[2]:
            raise ValueError(
                f"wavelengths shape mismatch: expected (n_wavelengths={arr.shape[2]},), got {wl.shape}. "
                "Ensure wavelengths are properly extracted from MODEST metadata."
            )

        I_norm = arr[:, 0, :]
        V_norm = arr[:, 1, :]
        I_mean_norm = np.mean(I_norm, axis=0)
        I_std_norm = np.std(I_norm, axis=0)
        V_mean_norm = np.mean(V_norm, axis=0)
        V_std_norm = np.std(V_norm, axis=0)

        I_mean = I_mean_norm
        I_std = I_std_norm
        V_mean = V_mean_norm
        V_std = V_std_norm
        if self.stokes_normalizer is not None and getattr(self.stokes_normalizer, "final_stats", None) is not None:
            stats_i = self.stokes_normalizer.final_stats.get("I", {})
            stats_v = self.stokes_normalizer.final_stats.get("V", {})
            mu_i = float(stats_i.get("mean", 0.0))
            sd_i = float(stats_i.get("std", 1.0))
            mu_v = float(stats_v.get("mean", 0.0))
            sd_v = float(stats_v.get("std", 1.0))
            I_mean = I_mean_norm * sd_i + mu_i
            I_std = I_std_norm * abs(sd_i)
            V_mean = V_mean_norm * sd_v + mu_v
            V_std = V_std_norm * abs(sd_v)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{display_name(self.model_name)} | Final Stokes Profiles: mean ± 1σ", fontsize=13, fontweight="bold")

        axes[0].plot(wl, I_mean, color="tab:orange", linewidth=1.8, label="Mean I")
        axes[0].fill_between(wl, I_mean - I_std, I_mean + I_std, color="tab:orange", alpha=0.25, label="±1σ")
        axes[0].set_ylabel(r"$I / I_\mathrm{c}$")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=9)

        axes[1].plot(wl, V_mean, color="tab:purple", linewidth=1.8, label="Mean V")
        axes[1].fill_between(wl, V_mean - V_std, V_mean + V_std, color="tab:purple", alpha=0.25, label="±1σ")
        axes[1].set_xlabel("Wavelength [Angstrom]")
        axes[1].set_ylabel(r"$V / I_\mathrm{c}$")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=9)

        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(self.out_dir / "stokes_mean_std_profiles.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def _write_metrics_csv(self) -> None:
        """Write the accumulated rows to ``metrics_summary.csv``."""
        if not self.metrics_rows:
            return
        metrics_path = self.out_dir / "metrics_summary.csv"
        fieldnames = [
            "model", "snapshot", "param", "logtau", "pairing", "n_points",
            "corr", "r2", "rmse", "rrmse", "mae", "nmae", "bias",
        ]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_rows)
        print(f"[{self.model_name}] Metrics saved to: {metrics_path}")

    def plot_image_panel(self, true_map: np.ndarray, pred_map: np.ndarray, p: str, od_eff: float) -> None:
        """Plot truth, prediction and their difference at one node (shared, quantile-clipped colour scale).

        Args:
            true_map: Reference map.
            pred_map: Prediction map, same shape.
            p: Quantity key.
            od_eff: The node.

        Examples:
            >>> plots.plot_image_panel(gt, pred, "Bz", -0.8)  # doctest: +SKIP
        """
        err_map = pred_map - true_map
        both = np.concatenate([true_map.ravel(), pred_map.ravel()])

        q = IMAGE_COLOR_QUANTILE
        if p in ("Vz", "Bz"):
            vmax = np.nanquantile(np.abs(both), 1.0 - q)
            vmin = -vmax
        else:
            vmin, vmax = np.nanquantile(both, [q, 1.0 - q])

        emax = np.nanquantile(np.abs(err_map.ravel()), 1.0 - q)
        param_cmap = self.param_cmaps.get(p, "viridis")

        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        im0 = ax[0].imshow(true_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
        ax[0].set_title(f"GT {p}")
        ax[0].axis("off")
        plt.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04).set_label(param_axis_label(p))

        im1 = ax[1].imshow(pred_map.T, origin="lower", cmap=param_cmap, vmin=vmin, vmax=vmax)
        ax[1].set_title(f"Pred {p}")
        ax[1].axis("off")
        plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04).set_label(param_axis_label(p))

        im2 = ax[2].imshow(err_map.T, origin="lower", cmap=self.error_cmap, vmin=-emax, vmax=emax)
        ax[2].set_title(f"Error {p}")
        ax[2].axis("off")
        plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04).set_label(
            param_axis_label(p, "Pred - GT")
        )

        fig.suptitle(f"Final Model | {display_name(self.model_name)} | Snapshot {self.label} | {p} @ log(tau)={od_eff:.2f}")
        fig.tight_layout()
        fig.savefig(self.out_dir / f"{p}_logtau_{od_eff:.2f}_images.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def plot_jointplot(self, true_map: np.ndarray, pred_map: np.ndarray, p: str, od_eff: float, tau_idx: int) -> dict[str, float] | None:
        """Scatter plot of prediction against truth at one node, with metrics on all paired pixels.

        The plot shows at most ``epoch_plot_scatter_samples`` points; the metrics use all.

        Args:
            true_map: Reference map.
            pred_map: Prediction map, same shape.
            p: Quantity key.
            od_eff: The node.
            tau_idx: Node index (seeds the plotting subsample).

        Returns:
            The metrics of :func:`compute_regression_metrics`, or ``None`` if no pair is finite.

        Examples:
            >>> m = plots.plot_jointplot(gt, pred, "T", 0.0, 30)  # doctest: +SKIP
        """
        x = true_map.ravel()
        y = pred_map.ravel()
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size == 0:
            return None

        metrics = compute_regression_metrics(x, y)

        if x.size > self.n_sample > 0:
            rng = np.random.default_rng(seed=tau_idx + 7)
            idx = rng.choice(x.size, size=self.n_sample, replace=False)
            x, y = x[idx], y[idx]

        lo, hi = np.nanquantile(np.concatenate([x, y]), [0.01, 0.99])
        g = sns.jointplot(
            x=x,
            y=y,
            kind="scatter",
            height=6,
            s=8,
            alpha=0.25,
            marginal_kws={"bins": 50, "fill": True},
        )
        g.ax_joint.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)
        g.ax_joint.set_xlabel(param_axis_label(p, "Ground truth"))
        g.ax_joint.set_ylabel(param_axis_label(p, "Prediction"))
        g.fig.suptitle(
            f"Final Model | {display_name(self.model_name)} | Snapshot {self.label} | {p} @ log(tau)={od_eff:.2f}\n"
            f"Corr={format_metric(metrics['corr'])}, R²={format_metric(metrics['r2'])}, "
            f"RRMSE={format_metric(metrics['rrmse'])}",
            y=1.02,
        )
        g.fig.tight_layout()
        g.fig.savefig(self.out_dir / f"{p}_logtau_{od_eff:.2f}_jointplot.png", dpi=170, bbox_inches="tight")
        plt.close(g.fig)
        return metrics

    def generate(
        self,
        pred_den: dict[str, np.ndarray],
        gt_den: dict[str, np.ndarray],
        stokes_input: np.ndarray | None = None,
        wavelengths: np.ndarray | None = None,
    ) -> None:
        """Make every panel and joint plot, the profile plot and the metrics CSV.

        Args:
            pred_den: Predicted ``{"T", "Vz", "Bz"}`` cubes in K, km/s and G.
            gt_den: MURaM cubes on the same nodes (resized to the prediction grid if needed).
            stokes_input: Model input for the profile plot, or ``None``.
            wavelengths: Its wavelength axis in A.

        Examples:
            >>> MuramDiagnosticPlots(cfg, "wfa_only", step=198).generate(pred, gt, x, wl)  # doctest: +SKIP
        """
        for od in self.ods:
            tau_idx = int(np.argmin(np.abs(self.logtau - od)))
            od_eff = float(self.logtau[tau_idx])

            for p in self.params:
                true_map = gt_den[p][:, :, tau_idx]
                pred_map = pred_den[p][:, :, tau_idx]

                # Align GT to pred resolution for fair pixel-wise diagnostics
                if true_map.shape != pred_map.shape:
                    true_map = self._resize_map_to_shape(true_map, pred_map.shape)

                self.plot_image_panel(true_map=true_map, pred_map=pred_map, p=p, od_eff=od_eff)
                metrics = self.plot_jointplot(true_map=true_map, pred_map=pred_map, p=p, od_eff=od_eff, tau_idx=tau_idx)
                if metrics is not None:
                    self.metrics_rows.append({
                        "model": self.model_name,
                        "snapshot": self.label,
                        "param": p,
                        "logtau": od_eff,
                        "pairing": "paired",
                        "n_points": int(metrics["n_points"]),
                        "corr": float(metrics["corr"]),
                        "r2": float(metrics["r2"]),
                        "rmse": float(metrics["rmse"]),
                        "rrmse": float(metrics["rrmse"]),
                        "mae": float(metrics["mae"]),
                        "nmae": float(metrics["nmae"]),
                        "bias": float(metrics["bias"]),
                    })

        if stokes_input is not None:
            self.plot_stokes_mean_std(stokes_input=stokes_input, wavelengths=wavelengths)

        self._write_metrics_csv()

    @staticmethod
    def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Bilinear resize of a 2-D map to ``target_shape``."""
        if arr2d.shape == target_shape:
            return arr2d
        t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()


# ------------------------------------------------------------------------------------------
# Multi-model comparison plots. Module-level (not methods) because they take N models' worth
# of data at once, unlike MuramDiagnosticPlots/ModestDiagnosticPlots which are built (or, for
# MODEST, iterate internally) one model at a time. Both classes call these after their normal
# per-model loop finishes, once >=2 models are available. Filenames intentionally do not
# collide with the existing per-model jointplot/images files -- callers put these under a
# separate "combined/" directory one level above the per-model output dirs.
# ------------------------------------------------------------------------------------------

def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2-D map to ``target_shape`` (unchanged if it already matches)."""
    if arr2d.shape == target_shape:
        return arr2d
    t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).cpu().numpy()


def plot_combined_jointplot_grid(
    true_map: np.ndarray,
    pred_by_model: dict[str, np.ndarray],
    param: str,
    title_suffix: str,
    tau_val: float,
    save_path: Path,
    model_order: tuple[str, ...] = MODEL_CANONICAL_ORDER,
    los: bool = False,
    max_points: int = 20000,
) -> None:
    """2 x 2 grid of prediction-against-truth scatter panels, one per model, sharing axis limits.

    With matching grids the panels pair pixels and show correlation, R^2 and RRMSE on all
    pairs; otherwise they compare matched quantiles and show the distribution metrics.

    Note:
        The grid has four panels, so with five models the fifth is dropped (strict xfail in
        ``tests/test_analysis.py``).

    Args:
        true_map: Reference map.
        pred_by_model: Prediction map per flavor.
        param: Quantity key.
        title_suffix: Context for the title.
        tau_val: The node.
        save_path: PNG path.
        model_order: Order of the panels.
        los: Use the line-of-sight symbols.
        max_points: Largest number of plotted points per panel.

    Examples:
        >>> plot_combined_jointplot_grid(gt, preds, "Bz", "MODEST - Sunspot", -0.8, Path("b.png"), los=True)  # doctest: +SKIP
    """
    plt.rcParams["font.family"] = "serif"
    models_present = [m for m in model_order if m in pred_by_model]
    if len(models_present) < 2:
        return

    x_all = np.asarray(true_map).ravel()
    x_finite = x_all[np.isfinite(x_all)]
    xy_by_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    paired_by_model: dict[str, bool] = {}
    metrics_text_by_model: dict[str, str] = {}
    lo_list, hi_list = [], []
    for m in models_present:
        y_raw = np.asarray(pred_by_model[m])
        if y_raw.shape == np.asarray(true_map).shape:
            # Same grid as ground truth -- pair pixel-by-pixel, as in the per-model jointplot.
            y_all = y_raw.ravel()
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            x_full, y_full = x_all[mask], y_all[mask]
            if x_full.size == 0:
                continue
            # Metrics are computed on the full paired population, before the plot-only
            # subsample below -- same convention as ModestDiagnosticPlots._plot_jointplot.
            metrics_reg = compute_regression_metrics(x_full, y_full)
            metrics_text_by_model[m] = (
                f"Corr={format_metric(metrics_reg['corr'])}\n"
                f"R²={format_metric(metrics_reg['r2'])}\n"
                f"RRMSE={format_metric(metrics_reg['rrmse'])}"
            )
            if x_full.size > max_points:
                rng = np.random.default_rng(seed=17)
                idx = rng.choice(x_full.size, size=max_points, replace=False)
                x, y = x_full[idx], y_full[idx]
            else:
                x, y = x_full, y_full
            paired_by_model[m] = True
        else:
            # Resolution mismatch (e.g. MODEST's upsampled prediction grid vs the native
            # ground-truth grid) -- fall back to matched-quantile comparison, same as
            # ModestDiagnosticPlots._plot_jointplot does for the single-model plots.
            y_finite = y_raw[np.isfinite(y_raw)].ravel()
            if x_finite.size == 0 or y_finite.size == 0:
                continue
            metrics_dist = compute_distribution_similarity_metrics(x_finite, y_finite)
            metrics_text_by_model[m] = (
                f"KS={format_metric(metrics_dist['ks'])}\n"
                f"W1q={format_metric(metrics_dist['w1_quantile'])}\n"
                f"JSD={format_metric(metrics_dist['jsd'])}\n"
                f"OVL={format_metric(metrics_dist['overlap'])}"
            )
            n = min(max_points, x_finite.size, y_finite.size)
            q = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
            x, y = np.quantile(x_finite, q), np.quantile(y_finite, q)
            paired_by_model[m] = False
        xy_by_model[m] = (x, y)
        both = np.concatenate([x, y])
        lo_list.append(np.nanquantile(both, 0.01))
        hi_list.append(np.nanquantile(both, 0.99))
    if not xy_by_model:
        return
    lo, hi = float(min(lo_list)), float(max(hi_list))

    # Same annotation style as notebooks/4-modest_data.ipynb's FOV-info box (plot_modest_continuum):
    # boxstyle='round,pad=0.5', facecolor='white', alpha=0.8.
    metrics_box_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for ax, m in zip(axes.ravel(), models_present):
        if m not in xy_by_model:
            ax.axis("off")
            continue
        x, y = xy_by_model[m]
        ax.scatter(x, y, s=6, alpha=0.25, color="tab:blue", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(display_name(m), fontsize=14)
        if paired_by_model[m]:
            ax.set_xlabel(param_axis_label(param, "Ground truth", los=los), fontsize=12)
            ax.set_ylabel(param_axis_label(param, "Prediction", los=los), fontsize=12)
        else:
            ax.set_xlabel(param_axis_label(param, "Ground truth (quantiles)", los=los), fontsize=12)
            ax.set_ylabel(param_axis_label(param, "Prediction (quantiles)", los=los), fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.25)
        if m in metrics_text_by_model:
            ax.text(
                0.05, 0.95, metrics_text_by_model[m], transform=ax.transAxes, fontsize=10,
                verticalalignment="top", bbox=metrics_box_props,
            )

    fig.suptitle(combined_plot_title(param, title_suffix, tau_val, los=los), fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_images_grid(
    true_map: np.ndarray,
    pred_by_model: dict[str, np.ndarray],
    param: str,
    title_suffix: str,
    tau_val: float,
    save_path: Path,
    model_order: tuple[str, ...] = MODEL_CANONICAL_ORDER,
    los: bool = False,
    transpose: bool = True,
    extent: tuple[float, float, float, float] | None = None,
    extent_labels: tuple[str, str] | None = None,
) -> None:
    """Truth, then one column per model with its prediction above and its error below.

    Truth and predictions share a colour scale set by the truth's own 2-98 % range (so
    over- or under-shoot shows as saturation); all error panels share one symmetric scale.
    Every image gets the same grid cell and the region's true aspect ratio. With ``extent``
    the truth panel and the first error panel carry spatial ticks (arcsec or Mm).

    Args:
        true_map: Reference map (resized to the prediction grid if needed).
        pred_by_model: Prediction map per flavor.
        param: Quantity key.
        title_suffix: Context for the title.
        tau_val: The node.
        save_path: PNG path.
        model_order: Order of the columns.
        los: Use the line-of-sight symbols.
        transpose: Draw the maps transposed (MURaM); MODEST maps use ``False``.
        extent: ``(left, right, bottom, top)`` in the drawn orientation, or ``None``.
        extent_labels: Axis names for the extent.

    Examples:
        >>> plot_combined_images_grid(gt, preds, "T", "Simulation step 198", 0.0, Path("t.png"))  # doctest: +SKIP
    """
    plt.rcParams["font.family"] = "serif"
    models_present = [m for m in model_order if m in pred_by_model]
    if len(models_present) < 2:
        return

    param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
    cmap = param_cmaps.get(param, "viridis")

    true_arr = np.asarray(true_map)
    # Ground truth and predictions can live on different native grids (e.g. MODEST's
    # upsampled prediction grid vs the native GT grid) -- resize GT to the prediction
    # resolution before computing errors, same convention as the per-model plots
    # (MuramDiagnosticPlots.generate's "Align GT to pred resolution").
    pred_shape = next((np.asarray(pred_by_model[m]).shape for m in models_present), true_arr.shape)
    if true_arr.shape != pred_shape:
        true_arr = _resize_map_to_shape(true_arr.astype(np.float32), pred_shape)

    gt = true_arr.T if transpose else true_arr
    panel_h, panel_w = gt.shape[:2]
    panel_aspect = (panel_h / panel_w) if panel_w else 1.0
    preds: dict[str, np.ndarray] = {}
    errs: dict[str, np.ndarray | None] = {}
    for m in models_present:
        p = np.asarray(pred_by_model[m])
        p = p.T if transpose else p
        preds[m] = p
        errs[m] = (p - gt) if p.shape == gt.shape else None

    # Color scale is calibrated to Ground Truth's OWN value range, not GT+predictions combined
    # -- GT should always render at full contrast, while a model whose predictions exceed that
    # range shows as saturated color, which is itself informative (it flags over/under-shoot
    # relative to the true physical range).
    q = IMAGE_COLOR_QUANTILE
    finite_gt = gt[np.isfinite(gt)]
    if finite_gt.size == 0:
        vmin, vmax = 0.0, 1.0
    elif param in ("Vz", "Bz"):
        vmax = float(np.nanquantile(np.abs(finite_gt), 1.0 - q))
        vmin = -vmax
    else:
        vmin, vmax = (float(v) for v in np.nanquantile(finite_gt, [q, 1.0 - q]))

    err_vals = [e[np.isfinite(e)] for e in errs.values() if e is not None]
    emax = float(np.nanquantile(np.abs(np.concatenate(err_vals)), 1.0 - q)) if err_vals else 1.0

    n_models = len(models_present)
    # Columns: [gt colorbar | spacer | GT | model_1 | ... | model_N | error colorbar]. GT and
    # the N model columns all share width_ratio 1.0 so they're identically sized; the colorbar
    # columns are narrow slivers that don't steal space from any image panel. The spacer is an
    # unused column that keeps the ground-truth panel's y tick labels from landing on top of
    # its colorbar once a spatial extent is drawn.
    n_cols = n_models + 4
    spacer_ratio = 0.30 if extent is not None else 0.02
    width_ratios = [0.05, spacer_ratio] + [1.0] * (n_models + 1) + [0.05]
    col_width_in = 2.6
    fig_width = col_width_in * (n_models + 1) + 1.0
    fig_height = col_width_in * panel_aspect * 2 + 0.9
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(2, n_cols, figure=fig, width_ratios=width_ratios, hspace=0.12, wspace=0.12)

    # imshow's own aspect would fight set_box_aspect once a physical extent is given, so the
    # box keeps the region's true ratio and the image is stretched to fill it.
    imshow_kw = {"origin": "lower", "extent": extent}
    if extent is not None:
        imshow_kw["aspect"] = "auto"

    def _spatial_ticks(ax) -> None:
        """Draw spatial ticks and axis names when an extent is given, else hide the ticks."""
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
            return
        ax.tick_params(labelsize=9)
        if extent_labels is not None:
            ax.set_xlabel(extent_labels[0], fontsize=11)
            ax.set_ylabel(extent_labels[1], fontsize=11)

    ax_gt = fig.add_subplot(gs[0, 2])
    ax_gt.set_box_aspect(panel_aspect)
    im_gt = ax_gt.imshow(gt, cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
    ax_gt.set_title("Ground Truth", fontsize=14)
    _spatial_ticks(ax_gt)

    cax_gt = fig.add_subplot(gs[0, 0])
    cbar_gt = plt.colorbar(im_gt, cax=cax_gt)
    cax_gt.yaxis.set_ticks_position("left")
    cax_gt.yaxis.set_label_position("left")
    cbar_gt.set_label(param_axis_label(param, los=los), fontsize=11)
    cbar_gt.ax.tick_params(labelsize=10)

    last_err_im = None
    for col, m in enumerate(models_present, start=3):
        ax_pred = fig.add_subplot(gs[0, col])
        ax_pred.set_box_aspect(panel_aspect)
        ax_pred.imshow(preds[m], cmap=cmap, vmin=vmin, vmax=vmax, **imshow_kw)
        ax_pred.set_title(display_name(m), fontsize=14)
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])

        ax_err = fig.add_subplot(gs[1, col])
        ax_err.set_box_aspect(panel_aspect)
        if errs[m] is not None:
            last_err_im = ax_err.imshow(errs[m], cmap="RdBu_r", vmin=-emax, vmax=emax, **imshow_kw)
        else:
            ax_err.text(0.5, 0.5, "shape\nmismatch", ha="center", va="center",
                        transform=ax_err.transAxes, fontsize=10)
        if col == 3:
            ax_pred.set_ylabel("", fontsize=13)
            # Once the panels carry spatial ticks the error row is named by its own colorbar
            # ("Pred - GT [unit]"), which frees the ylabel slot for the coordinate name.
            _spatial_ticks(ax_err)
            if extent is None:
                ax_err.set_ylabel("Error", fontsize=13)
        else:
            ax_err.set_xticks([])
            ax_err.set_yticks([])

    if last_err_im is not None:
        cax_err = fig.add_subplot(gs[1, n_cols - 1])
        cbar_err = plt.colorbar(last_err_im, cax=cax_err)
        cbar_err.set_label(param_axis_label(param, "Pred - GT", los=los), fontsize=11)
        cbar_err.ax.tick_params(labelsize=10)

    fig.suptitle(combined_plot_title(param, title_suffix, tau_val, los=los), fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_metric_vs_tau(
    tau_and_maps: list[tuple[float, np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    """RRMSE and correlation of truth against prediction at each node, sorted by node.

    A node whose maps have different shapes gives NaN instead of raising.

    Args:
        tau_and_maps: ``(tau, true_map, pred_map)`` triples in any order.

    Returns:
        ``{"tau", "rrmse", "corr"}`` arrays.

    Examples:
        >>> a = np.array([[1.0, 2.0], [3.0, 4.0]])
        >>> out = compute_metric_vs_tau([(0.0, a, a), (-1.0, a, a + 1.0), (0.5, a, a[:1])])
        >>> out["tau"].tolist(), round(float(out["rrmse"][0]), 3), bool(np.isnan(out["rrmse"][2]))
        ([-1.0, 0.0, 0.5], 0.4, True)
    """
    ordered = sorted(tau_and_maps, key=lambda t: t[0])
    taus = np.array([t[0] for t in ordered], dtype=float)
    rrmse = np.full(len(ordered), np.nan)
    corr = np.full(len(ordered), np.nan)
    for i, (_, true_map, pred_map) in enumerate(ordered):
        true_map = np.asarray(true_map)
        pred_map = np.asarray(pred_map)
        if true_map.shape != pred_map.shape:
            continue
        m = compute_regression_metrics(true_map, pred_map)
        rrmse[i] = m["rrmse"]
        corr[i] = m["corr"]
    return {"tau": taus, "rrmse": rrmse, "corr": corr}


_METRIC_VS_TAU_LABELS = {"rrmse": "RRMSE", "corr": "Corr"}


def plot_metric_vs_tau(
    param: str,
    metric_key: str,
    metrics_by_model: dict[str, dict[str, np.ndarray]],
    title_suffix: str,
    save_path: Path,
    model_order: tuple[str, ...] = MODEL_CANONICAL_ORDER,
    los: bool = False,
) -> None:
    """One-column figure of one metric (``"rrmse"`` or ``"corr"``) against log tau, one line per model.

    Args:
        param: Quantity key.
        metric_key: ``"rrmse"`` or ``"corr"``.
        metrics_by_model: Output of :func:`compute_metric_vs_tau` per flavor.
        title_suffix: Context for the title.
        save_path: PNG path (300 dpi).
        model_order: Order of the lines.
        los: Use the line-of-sight symbols.

    Examples:
        >>> plot_metric_vs_tau("Bz", "corr", metrics, "Simulation step 198", Path("c.png"))  # doctest: +SKIP
    """
    plt.rcParams["font.family"] = "serif"
    models_present = [m for m in model_order if m in metrics_by_model]
    if not models_present:
        return

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for m in models_present:
        d = metrics_by_model[m]
        ax.plot(d["tau"], d[metric_key], marker="o", ms=3, lw=1.2, label=display_name(m))

    ax.set_xlabel(r"$\log \tau$", fontsize=9)
    ax.set_ylabel(_METRIC_VS_TAU_LABELS.get(metric_key, metric_key), fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3)
    # Below the x-axis, not "best" -- with 4 lines "best" placement can land on top of the
    # data itself (e.g. correlation curves that cross near the legend's preferred corner).
    ax.legend(fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False)
    ax.set_title(combined_plot_title_no_tau(param, title_suffix, los=los), fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


class ModestDiagnosticPlots:
    """Per-model and combined comparisons of predictions with SPINOR on a MODEST scene or crop.

    :meth:`prepare_snapshot` loads and normalizes the input once; :meth:`run` predicts with
    every model, applies the optional sign inversions and temperature calibration, and
    writes maps, joint plots, ``metrics_summary.csv`` (the source of the paper's MODEST
    tables), ``metrics_by_field_strength.csv`` and the combined figures.

    Note:
        Paired metrics need the prediction on SPINOR's grid (``--downsample-prediction-input``);
        otherwise only distribution metrics are computed. The sign-inversion options exist
        only to undo the V x -1 applied on loading and are removed in reframe Phase 1d. The
        optional temperature calibration fits and evaluates a per-node bias on the same data.
    """
    def __init__(
        self,
        pipeline: AnalysisModelPipeline,
        modest_output_dir: Path,
        mhd_normalizer,
        stokes_normalizer,
        modest,
        modest_cache,
        args,
    ):
        """Store the pipeline, normalizers, MODEST loader and command-line options.

        Args:
            pipeline: An :class:`AnalysisModelPipeline`.
            modest_output_dir: Output folder for this scene or crop.
            mhd_normalizer: Fitted MHD normalizer.
            stokes_normalizer: Fitted Stokes normalizer.
            modest: A :class:`utils.modest_data.ModestData`.
            modest_cache: A :class:`utils.cache_manage.ModestDataCache`.
            args: Parsed options of ``modest_analysis.py``.
        """
        self.pipeline = pipeline
        self.modest_output_dir = modest_output_dir
        self.mhd_normalizer = mhd_normalizer
        self.stokes_normalizer = stokes_normalizer
        self.modest = modest
        self.modest_cache = modest_cache
        self.args = args

        self.modest_data = None
        self.modest_logtau = None
        self.modest_mhd_data = None
        self.modest_stokes_input = None
        self.pred_nx = None
        self.pred_ny = None
        self.n_tau_eff = None
        self.tau_indices = None
        self.metrics_rows: list[dict[str, float | str | int]] = []
        # Same comparisons as metrics_rows, split by the true |B_LOS| of each pixel.
        self.field_metrics_rows: list[dict[str, float | str | int]] = []
        self.modest_wavelength: np.ndarray | None = None

    @staticmethod
    def _resize_map_to_shape(arr2d: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        """Bilinear resize of a 2-D map to ``target_shape``."""
        if arr2d.shape == target_shape:
            return arr2d
        t = torch.from_numpy(arr2d).float().unsqueeze(0).unsqueeze(0)
        out = F.interpolate(t, size=target_shape, mode="bilinear", align_corners=False)
        return out.squeeze(0).squeeze(0).cpu().numpy()

    def _temperature_calibration_mode(self) -> str:
        """``"apply_fit"`` or ``"off"`` from the options."""
        mode = str(getattr(self.args, "temp_calibration_mode", "off")).lower()
        if mode not in {"off", "apply_fit"}:
            return "off"
        return mode

    def _temperature_calibration_clip_quantiles(self) -> tuple[float, float] | None:
        """Valid ``(q_low, q_high)`` from the options, else ``None``."""
        q = getattr(self.args, "temp_calibration_clip_quantiles", None)
        if q is None:
            return None
        if len(q) != 2:
            return None
        qlo = float(q[0])
        qhi = float(q[1])
        if not (0.0 <= qlo < qhi <= 1.0):
            return None
        return (qlo, qhi)

    def _temperature_calibration_min_samples(self) -> int:
        """Minimum pairs per node for a calibration fit (at least 10)."""
        return int(max(10, int(getattr(self.args, "temp_calibration_min_samples", 500))))

    def _temperature_calibration_path(self, model_type: str, out_root: Path) -> Path:
        """Calibration file of a flavor."""
        base_dir_raw = getattr(self.args, "temp_calibration_dir", None)
        if base_dir_raw:
            return Path(base_dir_raw) / model_type / "temperature_calibration.json"
        return out_root / "temperature_calibration.json"

    def _fit_temperature_affine_by_tau(
        self,
        model_type: str,
        matches: list[tuple[float, int, int]],
        true_cube: np.ndarray,
        pred_cube: np.ndarray,
    ) -> dict[str, Any]:
        """Per-node bias ``b = mean(T_true - T_pred)`` (K) on MODEST, with metrics before and after.

        The slope is fixed to 1. Diagnostics only: fitting and evaluating on the same pixels.
        """
        min_samples = self._temperature_calibration_min_samples()
        clip_q = self._temperature_calibration_clip_quantiles()
        coefficients: dict[str, dict[str, float | int]] = {}

        for tau_val, i_mod, i_pred in matches:
            if i_mod >= true_cube.shape[2] or i_pred >= pred_cube.shape[2]:
                continue
            true_map = np.asarray(true_cube[:, :, i_mod], dtype=np.float32)
            pred_map = np.asarray(pred_cube[:, :, i_pred], dtype=np.float32)
            if true_map.shape != pred_map.shape:
                true_map = self._resize_map_to_shape(true_map, pred_map.shape)

            x = true_map.ravel()  # GT
            y = pred_map.ravel()  # Pred
            m = np.isfinite(x) & np.isfinite(y)
            x = x[m]
            y = y[m]

            if x.size < min_samples:
                continue

            if clip_q is not None:
                qlo, qhi = clip_q
                x_lo, x_hi = np.quantile(x, [qlo, qhi])
                y_lo, y_hi = np.quantile(y, [qlo, qhi])
                m2 = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
                if np.count_nonzero(m2) >= min_samples:
                    x = x[m2]
                    y = y[m2]

            if x.size < min_samples:
                continue

            a = 1.0
            b = float(np.nanmean(x - y))

            metrics_before = compute_regression_metrics(x, y)
            metrics_after = compute_regression_metrics(x, y + b)
            coefficients[f"{float(tau_val):.6f}"] = {
                "tau_value": float(tau_val),
                "a": float(a),
                "b": float(b),
                "n_points": int(x.size),
                "corr_before": float(metrics_before["corr"]),
                "corr_after": float(metrics_after["corr"]),
                "rrmse_before": float(metrics_before["rrmse"]),
                "rrmse_after": float(metrics_after["rrmse"]),
            }

        return {
            "mode": "bias_per_tau",
            "model_type": model_type,
            "min_samples": int(min_samples),
            "clip_quantiles": list(clip_q) if clip_q is not None else None,
            "coefficients": coefficients,
        }

    @staticmethod
    def _save_temperature_calibration(path: Path, payload: dict[str, Any]) -> None:
        """Write a calibration JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def _load_temperature_calibration(path: Path) -> dict[str, Any] | None:
        """Read a calibration JSON, or ``None``."""
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _find_tau_coeff(coefficients: dict[str, dict[str, Any]], tau_val: float) -> dict[str, Any] | None:
        """Coefficients of the node ``tau_val`` (matched to 1e-6), or ``None``."""
        for key, coeff in coefficients.items():
            try:
                if np.isclose(float(key), float(tau_val), atol=1e-6, rtol=0.0):
                    return coeff
            except Exception:
                continue
        return None

    def _apply_temperature_calibration(
        self,
        pred_cube: np.ndarray,
        matches: list[tuple[float, int, int]],
        payload: dict[str, Any],
    ) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        """Add each node's bias to the predicted T cube and report what was applied."""
        coefficients = payload.get("coefficients", {}) if isinstance(payload, dict) else {}
        calibrated = np.array(pred_cube, copy=True)
        applied_by_pred_idx: dict[int, dict[str, float]] = {}

        for tau_val, _i_mod, i_pred in matches:
            if i_pred >= calibrated.shape[2]:
                continue
            coeff = self._find_tau_coeff(coefficients, tau_val)
            if coeff is None:
                continue
            b = float(coeff.get("b", 0.0))
            calibrated[:, :, i_pred] = calibrated[:, :, i_pred] + b
            applied_by_pred_idx[i_pred] = {
                "a": 1.0,
                "b": b,
                "n_points": int(coeff.get("n_points", 0)),
            }

        return calibrated, applied_by_pred_idx

    def _write_field_strength_metrics_csv(self, model_type: str, out_root: Path) -> None:
        """Write ``metrics_by_field_strength.csv``: the same comparisons split by SPINOR's |B_LOS| per pixel.

        The strong-field rows show extrapolation, not skill: the training steps 110 to 130 have
        many pixels below about 500 G, few above 950 G and none above 1500 G, while the sunspot
        reaches 3.7 kG.
        """
        rows = [r for r in self.field_metrics_rows if r.get("model") == model_type]
        if not rows:
            return
        field_path = out_root / "metrics_by_field_strength.csv"
        fieldnames = [
            "model", "param", "logtau", "field_bin", "bin_lo_G", "bin_hi_G",
            "n_points", "corr", "r2", "rmse", "rrmse", "mae", "nmae", "bias",
        ]
        with open(field_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{model_type}] Range-of-applicability metrics saved to: {field_path}")

    def _write_metrics_csv(self, model_type: str, out_root: Path) -> None:
        """Write a flavor's ``metrics_summary.csv`` and its field-strength companion."""
        self._write_field_strength_metrics_csv(model_type, out_root)
        rows = [r for r in self.metrics_rows if r.get("model") == model_type]
        if not rows:
            return
        metrics_path = out_root / "metrics_summary.csv"
        fieldnames = [
            "model", "param", "logtau", "comparison", "n_points", "n_true", "n_pred",
            "corr", "r2", "rmse", "rrmse", "mae", "nmae", "bias",
            "ks", "w1_quantile", "jsd", "overlap",
            "calibration_mode", "calibration_applied", "calibration_source",
            "calibration_a", "calibration_b", "calibration_n_fit",
        ]
        with open(metrics_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[{model_type}] Metrics saved to: {metrics_path}")

    def _resolve_tau_indices(self, indices: list[int] | None, n_tau: int) -> list[int]:
        """Valid node indices from the options (negative counts from the end); default first, middle, last."""
        if not indices:
            return sorted(set([0, n_tau // 2, n_tau - 1]))
        out = []
        for idx in indices:
            ridx = idx if idx >= 0 else n_tau + idx
            if 0 <= ridx < n_tau:
                out.append(ridx)
        return sorted(set(out))

    def prepare_snapshot(self, n_tau: int):
        """Load the MODEST scene or crop, SPINOR's atmosphere and the normalized model input.

        The input is ``prediction_stokes`` from ``ModestData.load_all`` (downsampled to the
        observed grid with ``--downsample-prediction-input``), after the optional spectral
        transforms of ``transform_modest_stokes_profiles``, normalized and stacked as I, V.

        Args:
            n_tau: Number of the model's nodes (caps the SPINOR nodes used).

        Examples:
            >>> diag.prepare_snapshot(n_tau=45)  # doctest: +SKIP
        """
        prediction_input_mode = "downsampled" if getattr(self.args, "downsample_prediction_input", False) else "upsampled"
        self.modest_data = self.modest.load_all(
            region_bounds=tuple(self.args.crop_bounds) if self.args.cropped_region else None,
            apply_mask=self.args.polarization_mask,
            cache=self.modest_cache,
            use_cache=not self.args.no_modest_cache,
            prediction_input_mode=prediction_input_mode,
        )
        self.modest_logtau = list(
            self.modest_data.get("tau_values", sorted(self.modest_data["spinor_atm"]["T"].keys()))
        )
        self.n_tau_eff = min(n_tau, len(self.modest_logtau))
        self.modest_mhd_data = {
            "T": np.stack(
                [self.modest_data["spinor_atm"]["T"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
            "Vz": np.stack(
                [self.modest_data["spinor_atm"]["Vlos"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
            "Bz": np.stack(
                [self.modest_data["spinor_atm"]["Blos"][t] for t in self.modest_logtau[:self.n_tau_eff]], axis=-1
            ).astype(np.float32),
        }
        # MODEST Stokes profiles are already continuum-normalized in the loader path.
        # Keep them as-is here and only apply the standard Stokes normalizer expected by the model.
        prediction_stokes = self.modest_data.get("prediction_stokes", self.modest_data["smoothed_stokes"])
        wl_raw = self.modest_data.get("wl", np.arange(prediction_stokes["I"].shape[-1]))
        self.modest_wavelength = np.asarray(
            wl_raw.value if hasattr(wl_raw, "value") else wl_raw,
            dtype=np.float64,
        )
        prediction_stokes = transform_modest_stokes_profiles(
            stokes=prediction_stokes,
            wavelength_angstrom=self.modest_wavelength,
            shift_positions=float(getattr(self.args, "modest_stokes_shift_positions", 0.0)),
            i_scale=float(getattr(self.args, "modest_stokes_i_scale", 1.0)),
            v_scale=float(getattr(self.args, "modest_stokes_v_scale", 1.0)),
            invert_direction=bool(getattr(self.args, "modest_stokes_invert_direction", False)),
        )
        self.pred_nx, self.pred_ny = prediction_stokes["I"].shape[:2]
        norm_stokes = self.stokes_normalizer.transform(prediction_stokes)
        I_flat = norm_stokes["I"].reshape(self.pred_nx * self.pred_ny, -1)
        V_flat = norm_stokes["V"].reshape(self.pred_nx * self.pred_ny, -1)
        self.modest_stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)
        self.tau_indices = self._resolve_tau_indices(self.args.tau_indices, self.n_tau_eff)
        print(f"\nMODEST optical depth nodes: {[float(v) for v in self.modest_logtau]}")
        gt_nx, gt_ny = self.modest_mhd_data["T"].shape[:2]
        print(f"MODEST prediction input mode: {prediction_input_mode}")
        print(f"MODEST prediction input grid: {self.pred_nx}x{self.pred_ny} (GT: {gt_nx}x{gt_ny})")

    def _filter_matches(self, matches):
        """Keep the node matches whose SPINOR index is among the selected ones."""
        if not self.tau_indices:
            return matches
        allowed = set(self.tau_indices)
        return [m for m in matches if m[1] in allowed]

    def _plot_imshows(
        self,
        true_2d: np.ndarray,
        pred_2d: np.ndarray,
        title: str,
        save_path: Path,
        param: str,
        transpose: bool = True,
        transpose_pred: bool | None = None,
    ):
        """Plot truth, prediction and their difference at one node (difference only if the grids match)."""
        gt = true_2d.T if transpose else true_2d
        pred_do_transpose = transpose if transpose_pred is None else transpose_pred
        pr = pred_2d.T if pred_do_transpose else pred_2d

        param_cmaps = {"T": "hot", "Vz": "bwr_r", "Bz": "PiYG"}
        cmap = param_cmaps.get(param, "viridis")

        vals = (
            np.concatenate([gt[np.isfinite(gt)], pr[np.isfinite(pr)]])
            if (np.isfinite(gt).any() and np.isfinite(pr).any())
            else np.array([0.0, 1.0])
        )
        q = IMAGE_COLOR_QUANTILE
        if param in ("Vz", "Bz"):
            vmax = np.quantile(np.abs(vals), 1.0 - q)
            vmin = -vmax
        else:
            vmin, vmax = np.quantile(vals, [q, 1.0 - q])

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        im0 = axes[0].imshow(gt, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f"Ground truth ({gt.shape[0]}x{gt.shape[1]})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04).set_label(param_axis_label(param, los=True))
        im1 = axes[1].imshow(pr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Prediction ({pr.shape[0]}x{pr.shape[1]})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04).set_label(param_axis_label(param, los=True))
        if gt.shape == pr.shape:
            er = pr - gt
            emax = np.quantile(np.abs(er[np.isfinite(er)]), 1.0 - q) if np.isfinite(er).any() else 1.0
            im2 = axes[2].imshow(er, origin="lower", cmap="RdBu_r", vmin=-emax, vmax=emax)
            axes[2].set_title("Error (Pred-GT)")
            plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04).set_label(
                param_axis_label(param, "Pred - GT", los=True)
            )
        else:
            axes[2].text(
                0.5,
                0.5,
                f"Error skipped\nshape mismatch\nGT={gt.shape}\nPred={pr.shape}",
                ha="center",
                va="center",
                transform=axes[2].transAxes,
            )
            axes[2].set_axis_off()
        fig.suptitle(title)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    def _plot_jointplot(
        self,
        true_2d: np.ndarray,
        pred_2d: np.ndarray,
        title: str,
        save_path: Path,
        param: str = "",
        max_points: int = 50000,
    ) -> tuple[dict[str, float], str] | None:
        """Scatter plot of prediction against SPINOR at one node, and its metrics.

        With matching grids, pixels are paired; otherwise matched quantiles are plotted and only
        the distribution metrics are computed.

        Returns ``(metrics, "paired_pixel" or "distribution_quantile")``, or ``None``.
        """
        if true_2d.shape == pred_2d.shape:
            x = true_2d.ravel()
            y = pred_2d.ravel()
            m = np.isfinite(x) & np.isfinite(y)
            x_pair, y_pair = x[m], y[m]
            if x_pair.size == 0:
                return None

            if x_pair.size > max_points:
                rng = np.random.default_rng(seed=17)
                idx = rng.choice(x_pair.size, size=max_points, replace=False)
                x_plot, y_plot = x_pair[idx], y_pair[idx]
            else:
                x_plot, y_plot = x_pair, y_pair

            metrics_reg = compute_regression_metrics(x_pair, y_pair)
            metrics_dist = compute_distribution_similarity_metrics(x_pair, y_pair)
            comparison = "paired_pixel"
            title_metrics = (
                f"paired | Corr={format_metric(metrics_reg['corr'])}, R²={format_metric(metrics_reg['r2'])}, "
                f"RRMSE={format_metric(metrics_reg['rrmse'])}"
            )
        else:
            x = true_2d[np.isfinite(true_2d)].ravel()
            y = pred_2d[np.isfinite(pred_2d)].ravel()
            if x.size == 0 or y.size == 0:
                return None

            n = min(max_points, x.size, y.size)
            q = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
            x_plot = np.quantile(x, q)
            y_plot = np.quantile(y, q)
            metrics_reg = {
                "n_points": np.nan,
                "corr": np.nan,
                "r2": np.nan,
                "rmse": np.nan,
                "rrmse": np.nan,
                "mae": np.nan,
                "nmae": np.nan,
                "bias": np.nan,
            }
            metrics_dist = compute_distribution_similarity_metrics(x, y)
            comparison = "distribution_quantile"
            title_metrics = (
                f"distribution | KS={format_metric(metrics_dist['ks'])}, "
                f"W1q={format_metric(metrics_dist['w1_quantile'])}, "
                f"JSD={format_metric(metrics_dist['jsd'])}, OVL={format_metric(metrics_dist['overlap'])}"
            )

        lo = float(min(np.min(x_plot), np.min(y_plot)))
        hi = float(max(np.max(x_plot), np.max(y_plot)))
        g = sns.jointplot(x=x_plot, y=y_plot, kind="scatter", s=8, alpha=0.25, height=6)
        g.ax_joint.plot([lo, hi], [lo, hi], "r--", lw=1.2)
        g.ax_joint.set_xlim(lo, hi)
        g.ax_joint.set_ylim(lo, hi)
        if comparison == "paired_pixel":
            g.ax_joint.set_xlabel(param_axis_label(param, "Ground truth", los=True))
            g.ax_joint.set_ylabel(param_axis_label(param, "Prediction", los=True))
        else:
            g.ax_joint.set_xlabel(param_axis_label(param, "Ground truth (quantiles)", los=True))
            g.ax_joint.set_ylabel(param_axis_label(param, "Prediction (quantiles)", los=True))
        g.fig.suptitle(
            f"{title}\n"
            f"{title_metrics}",
            y=1.02,
        )
        g.fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        out_metrics = {
            **metrics_reg,
            **metrics_dist,
        }
        return out_metrics, comparison

    def _plot_stokes_mean_std(self, model_type: str, out_root: Path) -> None:
        """Plot the mean and spread of the MODEST I and V input (only for a 2-channel input)."""
        if self.modest_stokes_input is None:
            return

        arr = np.asarray(self.modest_stokes_input, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1] != 2:
            return

        if self.modest_wavelength is None:
            raise ValueError(
                "MODEST wavelength metadata is required and must be set during prepare_snapshot(). "
                "Cannot use hardcoded defaults."
            )
        
        wl = np.asarray(self.modest_wavelength, dtype=np.float64)
        if wl.ndim != 1 or wl.size != arr.shape[2]:
            raise ValueError(
                f"MODEST wavelength shape mismatch: expected (n_wavelengths={arr.shape[2]},), got {wl.shape}. "
                "Ensure wavelength arrays are properly extracted from MODEST metadata."
            )

        I_norm = arr[:, 0, :]
        V_norm = arr[:, 1, :]
        I_mean_norm = np.mean(I_norm, axis=0)
        I_std_norm = np.std(I_norm, axis=0)
        V_mean_norm = np.mean(V_norm, axis=0)
        V_std_norm = np.std(V_norm, axis=0)

        I_mean = I_mean_norm
        I_std = I_std_norm
        V_mean = V_mean_norm
        V_std = V_std_norm
        stats = getattr(self.stokes_normalizer, "final_stats", None)
        if isinstance(stats, dict):
            stats_i = stats.get("I", {})
            stats_v = stats.get("V", {})
            mu_i = float(stats_i.get("mean", 0.0))
            sd_i = float(stats_i.get("std", 1.0))
            mu_v = float(stats_v.get("mean", 0.0))
            sd_v = float(stats_v.get("std", 1.0))
            I_mean = I_mean_norm * sd_i + mu_i
            I_std = I_std_norm * abs(sd_i)
            V_mean = V_mean_norm * sd_v + mu_v
            V_std = V_std_norm * abs(sd_v)

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        fig.suptitle(f"{display_name(model_type)} | MODEST Stokes Profiles: mean ± 1σ", fontsize=13, fontweight="bold")

        axes[0].plot(wl, I_mean, color="tab:orange", linewidth=1.8, label="Mean I")
        axes[0].fill_between(wl, I_mean - I_std, I_mean + I_std, color="tab:orange", alpha=0.25, label="±1σ")
        axes[0].set_ylabel(r"$I / I_\mathrm{c}$")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(loc="best", fontsize=9)

        axes[1].plot(wl, V_mean, color="tab:purple", linewidth=1.8, label="Mean V")
        axes[1].fill_between(wl, V_mean - V_std, V_mean + V_std, color="tab:purple", alpha=0.25, label="±1σ")
        axes[1].set_xlabel("Wavelength [Angstrom]")
        axes[1].set_ylabel(r"$V / I_\mathrm{c}$")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="best", fontsize=9)

        out_root.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_root / "stokes_mean_std_profiles.png", dpi=170, bbox_inches="tight")
        plt.close(fig)

    def run(self, model_configs, models):
        """Predict with every model and write all per-model and combined MODEST outputs.

        Args:
            model_configs: Entries of :meth:`AnalysisModelPipeline.get_model_configs`.
            models: Loaded models keyed like ``model_configs``.

        Examples:
            >>> diag.run(configs, models)  # doctest: +SKIP
        """
        calibration_mode = self._temperature_calibration_mode()
        print(f"Temperature calibration mode: {calibration_mode}")
        # Accumulated across the per-model loop below so the combined multi-model figures
        # (built after the loop) have every model's prediction available at once -- each
        # iteration otherwise overwrites pred_mhd/matches before the next model is loaded.
        pred_mhd_by_model: dict[str, dict[str, np.ndarray]] = {}
        matches_by_model: dict[str, list[tuple[float, int, int]]] = {}
        for name, model in models.items():
            model_type = model_configs[name]["experiment_key"]
            pred_tau = self.pipeline.get_model_logtau_values(model_configs[name])
            if pred_tau is None:
                print(f"[{model_type}] Predicted optical depth nodes: not found in experiment_results/config.")
                continue
            matches = self.pipeline.common_tau_matches([float(v) for v in self.modest_logtau], pred_tau)
            matches = self._filter_matches(matches)
            print(f"[{model_type}] Predicted optical depth nodes: {pred_tau}")
            print(f"[{model_type}] Common optical depth values (MODEST ∩ predicted): {[m[0] for m in matches]}")
            if not matches:
                print(f"[{model_type}] No common optical depths. Skipping plots.")
                continue
            pred_mhd = self.pipeline.predict_and_denormalize(
                model=model,
                stokes_input=self.modest_stokes_input,
                mhd_normalizer=self.mhd_normalizer,
                pred_nx=self.pred_nx,
                pred_ny=self.pred_ny,
                batch_size=self.args.inference_batch_size,
            )
            # Sign conventions for V_LOS and B_LOS are independent: velocity sign follows the
            # Doppler direction convention, field sign follows the circular-polarization
            # convention, and the two do not have to disagree with the reference in the same
            # way. The original flag flipped both together, which forces a choice that is
            # wrong for one of them whenever only one needs flipping. Per-parameter flags now
            # take precedence; the combined flag still applies to whichever is not set
            # individually, so existing commands keep working.
            invert_both = bool(getattr(self.args, "modest_pred_mhd_invert_sign", False))
            invert_vz = getattr(self.args, "modest_pred_vlos_invert_sign", None)
            invert_bz = getattr(self.args, "modest_pred_blos_invert_sign", None)
            invert_vz = invert_both if invert_vz is None else bool(invert_vz)
            invert_bz = invert_both if invert_bz is None else bool(invert_bz)
            if invert_vz:
                pred_mhd["Vz"] = -pred_mhd["Vz"]
            if invert_bz:
                pred_mhd["Bz"] = -pred_mhd["Bz"]
            if invert_vz or invert_bz:
                flipped = ", ".join(n for n, f in (("V_LOS", invert_vz), ("B_LOS", invert_bz)) if f)
                print(f"[{model_type}] Applied predicted MHD sign inversion for {flipped}.")
            out_root = self.modest_output_dir / model_type
            self._plot_stokes_mean_std(model_type=model_type, out_root=out_root)
            cal_path = self._temperature_calibration_path(model_type=model_type, out_root=out_root)

            applied_by_tau_idx: dict[int, dict[str, float]] = {}
            calibration_source = ""
            if calibration_mode == "apply_fit":
                cal_payload = self._load_temperature_calibration(cal_path)
                if cal_payload is None:
                    print(f"[{model_type}] No calibration file found — fitting now.")
                    cal_payload = self._fit_temperature_affine_by_tau(
                        model_type=model_type,
                        matches=matches,
                        true_cube=self.modest_mhd_data["T"],
                        pred_cube=pred_mhd["T"],
                    )
                    n_coeff = len(cal_payload.get("coefficients", {}))
                    if n_coeff > 0:
                        self._save_temperature_calibration(cal_path, cal_payload)
                        print(f"[{model_type}] Fitted and saved calibration ({n_coeff} taus) to: {cal_path}")
                    else:
                        print(f"[{model_type}] No calibration coefficients fitted (insufficient finite points). Proceeding without calibration.")
                        cal_payload = None
                else:
                    print(f"[{model_type}] Loaded existing calibration from: {cal_path}")
                if cal_payload is not None:
                    pred_mhd["T"], applied_by_tau_idx = self._apply_temperature_calibration(
                        pred_cube=pred_mhd["T"],
                        matches=matches,
                        payload=cal_payload,
                    )
                    calibration_source = str(cal_path)

            pred_mhd_by_model[model_type] = pred_mhd
            matches_by_model[model_type] = matches

            surface_dir = out_root / "surface"
            joint_dir = out_root / "jointplots"
            surface_dir.mkdir(parents=True, exist_ok=True)
            joint_dir.mkdir(parents=True, exist_ok=True)
            saved_any = False
            for param in ("T", "Vz", "Bz"):
                pred_cube = pred_mhd[param]
                true_cube = self.modest_mhd_data[param]
                for tau_val, i_mod, i_pred in matches:
                    if i_mod >= true_cube.shape[2] or i_pred >= pred_cube.shape[2]:
                        print(f"[{model_type}] Skip {param} @ tau={tau_val}: index out of bounds (mod={i_mod}, pred={i_pred}).")
                        continue
                    true_map = true_cube[:, :, i_mod]
                    pred_map = pred_cube[:, :, i_pred]
                    plot_title = (
                        f"{display_name(model_type)} | {param_latex(param, los=True)} | "
                        f"matched log(tau)={tau_val:.2f}"
                    )
                    if param == "T" and calibration_mode == "apply_fit":
                        if applied_by_tau_idx.get(i_pred) is not None:
                            plot_title += " | post-calibrated (apply_fit)"
                        else:
                            plot_title += " | raw prediction (apply_fit requested, no coeff)"
                    self._plot_imshows(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=plot_title,
                        save_path=surface_dir / f"{param}_tau_{tau_val:+.2f}_imshow.png",
                        param=param,
                        transpose=True,
                    )
                    metrics_out = self._plot_jointplot(
                        true_2d=true_map,
                        pred_2d=pred_map,
                        title=plot_title,
                        save_path=joint_dir / f"{param}_tau_{tau_val:+.2f}_jointplot.png",
                        param=param,
                    )
                    # Range-of-applicability breakdown: same comparison, split by the true
                    # |B_LOS| of each pixel, so the strong-field regime (where MURaM has few
                    # or no training pixels) is reported separately instead of being diluted
                    # by the weak-field majority. Reference strength is SPINOR's own B_LOS,
                    # so the split never depends on the model being evaluated.
                    if true_map.shape == pred_map.shape:
                        ref_bz = self.modest_mhd_data["Bz"][:, :, i_mod]
                        for row in compute_metrics_by_field_strength(true_map, pred_map, ref_bz):
                            self.field_metrics_rows.append({
                                "model": model_type,
                                "param": param,
                                "logtau": float(tau_val),
                                **row,
                            })

                    if metrics_out is not None:
                        metrics, comparison = metrics_out
                        cal_info = applied_by_tau_idx.get(i_pred) if param == "T" else None
                        self.metrics_rows.append({
                            "model": model_type,
                            "param": param,
                            "logtau": float(tau_val),
                            "comparison": comparison,
                            "n_points": int(metrics["n_points"]) if np.isfinite(metrics["n_points"]) else np.nan,
                            "n_true": int(metrics["n_true"]),
                            "n_pred": int(metrics["n_pred"]),
                            "corr": float(metrics["corr"]),
                            "r2": float(metrics["r2"]),
                            "rmse": float(metrics["rmse"]),
                            "rrmse": float(metrics["rrmse"]),
                            "mae": float(metrics["mae"]),
                            "nmae": float(metrics["nmae"]),
                            "bias": float(metrics["bias"]),
                            "ks": float(metrics["ks"]),
                            "w1_quantile": float(metrics["w1_quantile"]),
                            "jsd": float(metrics["jsd"]),
                            "overlap": float(metrics["overlap"]),
                            "calibration_mode": calibration_mode,
                            "calibration_applied": bool(cal_info is not None),
                            "calibration_source": calibration_source,
                            "calibration_a": float(cal_info["a"]) if cal_info is not None else np.nan,
                            "calibration_b": float(cal_info["b"]) if cal_info is not None else np.nan,
                            "calibration_n_fit": int(cal_info["n_points"]) if cal_info is not None else np.nan,
                        })
                    saved_any = True
            if saved_any:
                print(f"[{model_type}] Surface images saved to: {surface_dir}")
                print(f"[{model_type}] Jointplots saved to: {joint_dir}")
                self._write_metrics_csv(model_type=model_type, out_root=out_root)
            else:
                print(f"[{model_type}] No images were saved.")

        # Combined multi-model comparison figures (Ground Truth + one prediction/error column
        # per model), one per (param, tau) -- only meaningful with 2+ models, matching the
        # auto-trigger convention already used for cross-model comparison elsewhere
        # (tools/run_nicole_synthesis.sh's steps 4-5).
        if len(pred_mhd_by_model) >= 2:
            combined_dir = self.modest_output_dir / "combined"
            # Only compare taus present in every model's own matches -- in this project all
            # four checkpoints share one 45-level tau grid, so this is normally just the full
            # set, but a model with a different predicted grid would otherwise silently
            # misalign columns.
            common_tau_vals = None
            for m_matches in matches_by_model.values():
                taus = {round(float(t), 6) for t, _, _ in m_matches}
                common_tau_vals = taus if common_tau_vals is None else (common_tau_vals & taus)
            common_tau_vals = sorted(common_tau_vals or set())

            any_matches = next(iter(matches_by_model.values()))
            i_mod_by_tau = {round(float(t), 6): i_mod for t, i_mod, _ in any_matches}

            # Helioprojective extent of the crop, so the maps carry a spatial scale that can be
            # matched against the full-scene figure. Derived from the NATIVE crop bounds, which
            # is why it holds whether or not the predictions were downsampled back to 0.32".
            scene = getattr(self.modest, "continuum", None)
            if scene is not None:
                scene_shape = (int(scene.shape[0]), int(scene.shape[1]))
                if getattr(self.args, "cropped_region", False):
                    crop_bounds = tuple(int(v) for v in self.args.crop_bounds)
                else:
                    crop_bounds = (0, scene_shape[0], 0, scene_shape[1])
                map_extent, map_extent_labels = modest_crop_extent(crop_bounds, scene_shape)
            else:
                map_extent, map_extent_labels = None, None

            for param in ("T", "Vz", "Bz"):
                true_cube = self.modest_mhd_data[param]
                for tau_r in common_tau_vals:
                    i_mod = i_mod_by_tau[tau_r]
                    if i_mod >= true_cube.shape[2]:
                        continue
                    true_map = true_cube[:, :, i_mod]
                    pred_by_model: dict[str, np.ndarray] = {}
                    for m, m_matches in matches_by_model.items():
                        i_pred = next((ip for t, im, ip in m_matches if round(float(t), 6) == tau_r), None)
                        pred_cube = pred_mhd_by_model[m][param]
                        if i_pred is None or i_pred >= pred_cube.shape[2]:
                            continue
                        pred_by_model[m] = pred_cube[:, :, i_pred]
                    region_label = str(getattr(self.args, "crop_label", "") or "whole").replace("_", " ")
                    title_context = f"MODEST - {region_label.capitalize()}"
                    plot_combined_jointplot_grid(
                        true_map=true_map,
                        pred_by_model=pred_by_model,
                        param=param,
                        title_suffix=title_context,
                        tau_val=tau_r,
                        save_path=combined_dir / f"{param}_tau_{tau_r:+.2f}_combined_jointplot.png",
                        los=True,
                    )
                    plot_combined_images_grid(
                        true_map=true_map,
                        pred_by_model=pred_by_model,
                        param=param,
                        title_suffix=title_context,
                        tau_val=tau_r,
                        save_path=combined_dir / f"{param}_tau_{tau_r:+.2f}_combined_images.png",
                        # MODEST maps are (y, x); drawn untransposed so Solar-X runs horizontally,
                        # matching the full-scene and regions-overview figures.
                        transpose=False,
                        extent=map_extent,
                        extent_labels=map_extent_labels,
                        los=True,
                    )

            # Metric-vs-log(tau) summary curves, one line per model. MODEST's real ground truth
            # only has a handful of native optical-depth nodes (the SPINOR inversion's own
            # levels), so each line has as many points as that model's own matches -- not the
            # 45-level grid MURaM's version gets.
            for param in ("T", "Vz", "Bz"):
                true_cube = self.modest_mhd_data[param]
                metrics_by_model = {}
                for m, m_matches in matches_by_model.items():
                    pred_cube = pred_mhd_by_model[m][param]
                    tau_and_maps = [
                        (float(tau_val), true_cube[:, :, i_mod], pred_cube[:, :, i_pred])
                        for tau_val, i_mod, i_pred in m_matches
                        if i_mod < true_cube.shape[2] and i_pred < pred_cube.shape[2]
                    ]
                    metrics_by_model[m] = compute_metric_vs_tau(tau_and_maps)
                region_label = str(getattr(self.args, "crop_label", "") or "whole").replace("_", " ")
                title_context = f"MODEST - {region_label.capitalize()}"
                for metric_key in ("rrmse", "corr"):
                    plot_metric_vs_tau(
                        param=param,
                        metric_key=metric_key,
                        metrics_by_model=metrics_by_model,
                        title_suffix=title_context,
                        save_path=combined_dir / f"{param}_{metric_key}_vs_tau.png",
                        los=True,
                    )
            print(f"Combined multi-model figures saved to: {combined_dir}")
