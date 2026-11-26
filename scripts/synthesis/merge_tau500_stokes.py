#!/usr/bin/env python3
"""Merge per-chunk tau_500 NICOLE Stokes into the full-frame cube (issue #5).

Assembles the row-chunks written by generate_tau500_stokes.py for one step
into the full (nx, ny, N_WL, 4) Stokes cube and the full (nx, ny, nz)
tau_500 atmosphere, validating that the chunks tile the frame exactly (no
gaps, no overlaps) and contain no NaNs before writing the merged outputs.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig

CHUNK_RE = re.compile(r"chunk_(\d+)_(\d+)_(\d+)_stokes\.npy$")


# Physical range of the continuum in HSRA units: MURaM's darkest and brightest pixels sit
# around 0.69 and 1.46 at the continuum points. Values outside mean NICOLE failed silently
# (e.g. the negative continua it returns when the local-frame field slots are filled).
CONTINUUM_MIN, CONTINUUM_MAX = 0.3, 2.0


def count_bad_continua(stokes: np.ndarray) -> int:
    """Number of pixels whose continuum (mean I of the first 4 wavelengths) is outside the physical range.

    Args:
        stokes: ``(..., n_wl, 4)`` Stokes cube, I first, in HSRA continuum units.

    Returns:
        The count of pixels with a continuum below ``CONTINUUM_MIN`` or above
        ``CONTINUUM_MAX`` (NaN counts as bad).

    Examples:
        >>> cube = np.ones((2, 3, 5, 4)); cube[0, 1, :, 0] = -12.0
        >>> count_bad_continua(cube)
        1
    """
    ic = np.asarray(stokes[..., :4, 0]).mean(-1)
    return int(np.sum(~((ic >= CONTINUUM_MIN) & (ic <= CONTINUUM_MAX))))


def main():
    """Command-line entry point (see the module docstring for the options and outputs).

    Examples:
        ::

            python scripts/synthesis/merge_tau500_stokes.py --step 140 --chunk-dir chunks/ --output-dir data/muram-simulation/
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--chunk-dir", type=Path, required=True,
                   help="Directory holding chunk_{step}_{r0}_{r1}_stokes.npy / _atmos.npz")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write the merged stokes_{step}_nicole_tau500.npy / atmos")
    args = p.parse_args()

    train_cfg = TrainingConfig()
    nx, ny = train_cfg.nx, train_cfg.ny

    chunks = []
    for f in sorted(args.chunk_dir.glob(f"chunk_{args.step}_*_stokes.npy")):
        m = CHUNK_RE.search(f.name)
        if not m or int(m.group(1)) != args.step:
            continue
        r0, r1 = int(m.group(2)), int(m.group(3))
        chunks.append((r0, r1, f))
    if not chunks:
        sys.exit(f"No chunks found for step {args.step} in {args.chunk_dir}")
    chunks.sort()

    # Coverage check: chunks must tile [0, nx) exactly.
    covered = np.zeros(nx, dtype=int)
    for r0, r1, _ in chunks:
        covered[r0:r1] += 1
    if not np.all(covered == 1):
        gaps = np.where(covered == 0)[0]
        overlaps = np.where(covered > 1)[0]
        sys.exit(f"Chunks do not tile [0,{nx}) exactly. gaps={gaps.tolist()[:10]} "
                 f"overlaps={overlaps.tolist()[:10]}")

    n_wl = np.load(chunks[0][2], mmap_mode="r").shape[2]
    stokes_full = np.empty((nx, ny, n_wl, 4), dtype=np.float64)
    logtau = None
    # Every per-pixel array of the chunk files is stitched (T, Vz, v_los, Bz, Bx, By, z_km,
    # whatever the generator wrote), so new keys pass through without editing this script.
    atmos_full: dict[str, np.ndarray] = {}

    for r0, r1, f in chunks:
        s = np.load(f)
        if s.shape != (r1 - r0, ny, n_wl, 4):
            sys.exit(f"Chunk {f.name} has shape {s.shape}, expected {(r1 - r0, ny, n_wl, 4)}")
        stokes_full[r0:r1] = s
        a = np.load(f.with_name(f.name.replace("_stokes.npy", "_atmos.npz")))
        if logtau is None:
            logtau = a["logtau"]
            nz = len(logtau)
            keys = [k for k in a.files if a[k].ndim == 3 and a[k].shape[1:] == (ny, nz)]
            atmos_full = {k: np.empty((nx, ny, nz), dtype=np.float64) for k in keys}
        # Every chunk must carry the same arrays: a chunk left over from an older generator
        # (no v_los, Bx, By) would otherwise drop those keys silently or mix sign conventions.
        chunk_keys = {k for k in a.files if a[k].ndim == 3 and a[k].shape[1:] == (ny, nz)}
        if chunk_keys != set(atmos_full):
            sys.exit(f"Chunk {f.name} has arrays {sorted(chunk_keys)} but the first chunk has "
                     f"{sorted(atmos_full)}: chunks from different generator versions? Regenerate them.")
        for k in atmos_full:
            atmos_full[k][r0:r1] = a[k]

    n_nan = int(np.isnan(stokes_full).sum())
    if n_nan:
        sys.exit(f"Merged Stokes contains {n_nan} NaNs -- refusing to write")
    n_bad = count_bad_continua(stokes_full)
    if n_bad:
        sys.exit(f"Merged Stokes has {n_bad} pixels with a continuum outside "
                 f"[{CONTINUUM_MIN}, {CONTINUUM_MAX}] (HSRA units) -- refusing to write. "
                 "NICOLE returned garbage for them; check the model writer and the chunks.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stokes_out = args.output_dir / f"stokes_{args.step}_nicole_tau500.npy"
    atmos_out = args.output_dir / f"atmos_{args.step}_tau500.npz"
    np.save(stokes_out, stokes_full)
    np.savez(atmos_out, logtau=logtau, step=args.step, **atmos_full)

    cont = np.mean([stokes_full[..., 0][..., :5].mean(), stokes_full[..., 0][..., -5:].mean()])
    print(f"Merged {len(chunks)} chunks for step {args.step}")
    print(f"  stokes -> {stokes_out}  shape {stokes_full.shape}")
    print(f"  atmos  -> {atmos_out}   ({'/'.join(atmos_full)} on {len(logtau)}-level tau_500 grid)")
    print(f"  mean continuum I ~ {cont:.4f}   NaNs: {n_nan}")


if __name__ == "__main__":
    main()
