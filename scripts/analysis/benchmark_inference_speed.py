#!/usr/bin/env python3
"""Benchmark MUISCA inference against a NICOLE forward synthesis, for the paper's speed table.

Every column is measured on the machine the script runs on, under conditions stated in the output:

  - NICOLE: one forward synthesis per pixel, run sequentially through the same NicoleRunner the
    synthesis validation uses (one NICOLE process at a time, so a single core), on real MUISCA
    predictions read from an existing predictions.h5. The time includes writing the model and
    reading back the profiles, as in the pipeline. Workdirs go to a temporary directory, so the
    stored syntheses are not touched.
  - MUISCA: the trained network applied to a full map of NX x NY pixels in a single batch, on one
    CPU thread, on N CPU threads, and on the GPU. The GPU time includes copying the input to the
    device and the output back.

Writes output/paper_tables/speed_benchmark.md (the LaTeX table plus the measurement conditions)
and speed_benchmark.json (raw timings).

Usage:
    python scripts/analysis/benchmark_inference_speed.py [--nicole-pixels 60]
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import socket
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.analysis import AnalysisModelPipeline  # noqa: E402
from utils.synthesis import NicoleRunner, SynthesisConfig  # noqa: E402

EXPERIMENT_ROOT = "experiment_110_to_130-step_size_10-normal"
MODEL_TYPE = "no_physics"   # the architecture, and hence the cost, is the same for every flavor
PREDICTIONS_H5 = (ROOT / "output" / "synthesis" / EXPERIMENT_ROOT / "modest" / MODEL_TYPE
                  / "sunspot" / "predictions.h5")
NX, NY = 480, 480           # one MURaM-sized map
N_WL, N_CHANNELS = 112, 2   # Hinode/SOT-SP grid, Stokes I and V


def cpu_model() -> str:
    """CPU model name from ``/proc/cpuinfo``, without the core-count suffix.

    Examples:
        >>> isinstance(cpu_model(), str)
        True
    """
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            # "AMD EPYC 9354P 32-Core Processor" -> "AMD EPYC 9354P"
            return re.sub(r"\s*\d+-Core Processor\s*$", "", line.split(":", 1)[1].strip())
    return platform.processor()


def time_nicole(n_pixels: int) -> list[float]:
    """Seconds per pixel of sequential single-pixel NICOLE syntheses of real MUISCA predictions.

    Args:
        n_pixels: Number of pixels from ``PREDICTIONS_H5``.

    Returns:
        One time per pixel, including writing the model and reading the profile.

    Examples:
        >>> times = time_nicole(10)  # doctest: +SKIP
    """
    with h5py.File(PREDICTIONS_H5, "r") as h5:
        pixels = [tuple(int(v) for v in p) for p in h5["pixels"][:n_pixels]]
        logtau = h5["logtau"][...]
        T, Vz, Bz = h5["T"][:n_pixels], h5["Vz"][:n_pixels], h5["Bz"][:n_pixels]
        attrs = dict(h5.attrs)

    tmp = Path(tempfile.mkdtemp(prefix="nicole_bench_"))
    try:
        cfg = SynthesisConfig(source=str(attrs.get("source", "modest")), experiment_root=EXPERIMENT_ROOT,
                              model_type=MODEL_TYPE, region_label=str(attrs.get("region_label", "sunspot")),
                              output_root=tmp)
        runner = NicoleRunner(cfg)
        per_pixel = []
        for k, (ix, iy) in enumerate(pixels):
            t0 = time.perf_counter()
            workdir = runner.prepare_workdir(ix=ix, iy=iy, logtau=logtau, T=T[k], v_los_kms=Vz[k],
                                             b_long_G=Bz[k], gt_pressure=None)
            runner.run_pixel(workdir)
            per_pixel.append(time.perf_counter() - t0)
        return per_pixel
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_model(device: torch.device) -> torch.nn.Module:
    """Load the benchmarked flavor (``MODEL_TYPE``) on ``device`` in evaluation mode.

    Args:
        device: CPU or GPU.

    Returns:
        The model.

    Examples:
        >>> model = load_model(torch.device("cpu"))  # doctest: +SKIP
    """
    pipeline = AnalysisModelPipeline(device=device, output_dir=Path(tempfile.gettempdir()) / "bench_out",
                                     experiment_root=EXPERIMENT_ROOT, experiments_base_dir=None)
    _, models, _ = pipeline.prepare_models([MODEL_TYPE])
    return next(iter(models.values())).eval()


def time_muisca(device: torch.device, n_threads: int | None, repeats: int) -> list[float]:
    """Seconds per full 480 x 480 map in one batch (input from and output to the CPU).

    Args:
        device: CPU or GPU.
        n_threads: CPU threads (``None`` leaves torch's setting).
        repeats: Number of timed passes after one warm-up.

    Returns:
        One time per pass.

    Examples:
        >>> times = time_muisca(torch.device("cpu"), 1, 3)  # doctest: +SKIP
    """
    if n_threads is not None:
        torch.set_num_threads(n_threads)
    model = load_model(device)
    x = torch.randn(NX * NY, N_CHANNELS, N_WL)
    times = []
    with torch.no_grad():
        model(x[:4096].to(device))   # warm-up
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x.to(device)).cpu()
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return times


def fmt_time(seconds: float) -> str:
    r"""A duration as LaTeX in ms, s or h.

    Args:
        seconds: The duration in s.

    Examples:
        >>> fmt_time(0.0123), fmt_time(4.2), fmt_time(7200.0)
        ('$12.3$~ms', '$4.2$~s', '$\\sim\\!2.0$~h')
    """
    if seconds >= 3600:
        return f"$\\sim\\!{seconds / 3600:.1f}$~h"
    if seconds >= 1:
        return f"${seconds:.1f}$~s"
    return f"${seconds * 1e3:.3g}$~ms"


def fmt_speedup(factor: float) -> str:
    r"""A speed-up factor as LaTeX, in scientific notation above 10.

    Args:
        factor: The speed-up.

    Examples:
        >>> fmt_speedup(3.0), fmt_speedup(25000.0)
        ('$3\\times$', '$\\sim\\!2.5\\times10^{4}$')
    """
    if factor < 10:
        return f"${factor:.0f}\\times$"
    exponent = int(np.floor(np.log10(factor)))
    return f"$\\sim\\!{factor / 10**exponent:.1f}\\times10^{{{exponent}}}$"


def main() -> None:
    """Command-line entry point (see the module docstring for the options and outputs).

    Examples:
        ::

            python scripts/analysis/benchmark_inference_speed.py --nicole-pixels 60 --threads 32
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nicole-pixels", type=int, default=60)
    parser.add_argument("--threads", type=int, default=32, help="multi-threaded CPU column")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "paper_tables")
    args = parser.parse_args()

    n_map = NX * NY
    nicole = time_nicole(args.nicole_pixels)
    nicole_px = statistics.median(nicole)
    print(f"NICOLE: median {nicole_px * 1e3:.1f} ms/pixel over {len(nicole)} pixels")

    columns = [("NICOLE", "(1 core)", nicole_px * n_map, nicole_px)]
    runs = {"nicole_s_per_pixel": nicole}
    for label, sub, device, threads, repeats in [
        ("MUISCA", "(CPU, 1 thread)", torch.device("cpu"), 1, 3),
        ("MUISCA", f"(CPU, {args.threads} threads)", torch.device("cpu"), args.threads, 5),
        ("MUISCA", "(GPU)", torch.device("cuda"), None, 20),
    ]:
        if device.type == "cuda" and not torch.cuda.is_available():
            print("No GPU available; skipping the GPU column.")
            continue
        times = time_muisca(device, threads, repeats)
        per_map = statistics.median(times)
        columns.append((label, sub, per_map, per_map / n_map))
        runs[f"muisca_{device.type}_{threads or 'gpu'}_s_per_map"] = times
        print(f"MUISCA {sub}: median {per_map:.3f} s/map, {per_map / n_map * 1e3:.4f} ms/pixel")

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    host = {"hostname": socket.gethostname(), "cpu": cpu_model(), "gpu": gpu_name,
            "torch": torch.__version__, "map_pixels": n_map, "nicole_pixels": len(nicole)}

    lines = [
        "\\begin{table}[!htbp]", "\\centering", "\\footnotesize",
        "\\caption{Inference speed comparison between MUISCA's learned inverse mapping (Stokes $\\to$ "
        "atmosphere) and NICOLE's forward synthesis (atmosphere $\\to$ Stokes), per pixel and for a map of "
        f"${NX} \\times {NY}$ pixels, all measured on the same node ({host['cpu']} CPU, {gpu_name} GPU). "
        "NICOLE synthesizes one pixel at a time on a single core; MUISCA is applied to the whole map in a "
        f"single batch, on one and on {args.threads} CPU threads and on the GPU, including the copies to and "
        "from the GPU. The comparison is carried out against a single NICOLE synthesis call, not against a "
        "full NICOLE inversion, since the latter would require many calls per pixel inside an iterative fit "
        "and would therefore be substantially slower.}",
        "\\label{tab:speed_comparison}",
        "\\begin{tabular}{l" + "c" * len(columns) + "}", "\\hline",
        " & " + " & ".join(c[0] for c in columns) + " \\\\",
        " & " + " & ".join(c[1] for c in columns) + " \\\\", "\\hline",
        "Time/pixel & " + " & ".join(fmt_time(c[3]) for c in columns) + " \\\\",
        "Time/map & " + " & ".join(fmt_time(c[2]) for c in columns) + " \\\\",
        "Speedup & " + " & ".join(fmt_speedup(nicole_px / c[3]) for c in columns) + " \\\\",
        "\\hline", "\\end{tabular}", "\\end{table}",
    ]
    md = [
        "# Inference speed benchmark", "",
        f"Generated by `scripts/analysis/benchmark_inference_speed.py` on {datetime.now():%Y-%m-%d %H:%M}.", "",
        f"- Host: `{host['hostname']}`, CPU {host['cpu']}, GPU {gpu_name}, torch {host['torch']}",
        f"- NICOLE: median over {len(nicole)} sequential syntheses of real MUISCA predictions "
        f"(`{PREDICTIONS_H5.relative_to(ROOT)}`), including model writing and profile reading",
        f"- MUISCA (`{MODEL_TYPE}`): median over repeated passes on a full {NX}x{NY} map in one batch", "",
        "```latex", *lines, "```", "",
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "speed_benchmark.md").write_text("\n".join(md), encoding="utf-8")
    (args.output_dir / "speed_benchmark.json").write_text(json.dumps({"host": host, "runs": runs}, indent=2))
    print(f"Written to {args.output_dir / 'speed_benchmark.md'}")


if __name__ == "__main__":
    main()
