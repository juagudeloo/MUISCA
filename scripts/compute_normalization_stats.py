import os
import sys
sys.path.append("/scratchsan/observatorio/juagudeloo/MUISCA/")

from pathlib import Path
import json
import shutil
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm

from utils.muram_data import StokesData
from utils.normalizer import MhdNormalizer, StokesNormalizer
from utils.cache_manage import MuramDataCache
from scripts.base_training import (TrainingConfig, build_cache_config_signature, load_source_arrays,
                                   require_one_velocity_convention)


def _compute_fixed_global_ic(
    stokes_data_dir: Path,
    start_step: int,
    end_step: int,
    cont_indices: list[int],
) -> tuple[float, dict[int, float]]:
    """Mean continuum intensity over early legacy steps, for the legacy ``fixed_global`` normalization.

    Only used with ``data_source="muram_legacy"``; the NICOLE profiles are already normalized
    to the HSRA continuum.

    Returns:
        ``(fixed_ic, ic_per_step)`` in the legacy cube's intensity units.

    Raises:
        ValueError: For an empty step range or index list, or a non-positive I_c.
    """
    if end_step < start_step:
        raise ValueError(f"ic_end_step ({end_step}) must be >= ic_start_step ({start_step})")

    per_step_ic: dict[int, float] = {}
    cont_indices_np = np.asarray(cont_indices, dtype=int)
    if cont_indices_np.ndim != 1 or cont_indices_np.size == 0:
        raise ValueError("ic_cont_indices must be a non-empty list of indices")

    steps = list(range(start_step, end_step + 1))
    print(
        f"\nComputing fixed global I_c from early steps [{start_step}, {end_step}] "
        f"using continuum indices {cont_indices_np.tolist()}"
    )

    for step in tqdm(steps, desc="Computing early-step I_c"):
        stokes = StokesData(
            data_dir=stokes_data_dir,
            step=step,
            wavelength_range=(6300.5, 6303.5),
            wavelength_step=0.01,
        )
        stokes.load_stokes()
        ic_step = float(stokes.data["I"][:, :, cont_indices_np].mean(axis=2).mean())
        if not np.isfinite(ic_step) or ic_step <= 0:
            raise ValueError(f"Invalid I_c at step {step}: {ic_step}")
        per_step_ic[step] = ic_step

    fixed_ic = float(np.mean(list(per_step_ic.values())))
    if not np.isfinite(fixed_ic) or fixed_ic <= 0:
        raise ValueError(f"Computed fixed I_c is invalid: {fixed_ic}")

    print(f"Fixed global I_c (mean over early steps): {fixed_ic:.6e}")
    return fixed_ic, per_step_ic

def compute_normalization_stats(
    min_step: int = 60,
    max_step: int = 223,
    steps: list[int] | None = None,
    save_interval: int = 20,
    resume_from: str = None,
    logtau_values: list[float] | None = None,
    logtau_min: float = -3.0,
    logtau_max: float = 1.4,
    logtau_step: float = 0.1,
    use_cache: bool = True,
    cache_dir: str = "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache",
    stokes_ic_mode: str = "fixed_global",
    ic_start_step: int = 70,
    ic_end_step: int = 80,
    ic_cont_indices: list[int] | None = None,
    clean_start: bool = False,
    purge_cache: bool = False,
    data_source: str = "nicole_tau500",
    data_path: str | Path = "/scratchsan/observatorio/juagudeloo/MUISCA/data/",
    output_dir: str | Path | None = None,
    kappa_path: str | Path | None = None,
    lsf_path: str | Path | None = None,
):
    """Accumulate the MHD and Stokes normalization statistics over the training steps and save them.

    Each step is read through the same pipeline as training (cache, LSF, resampling), its
    statistics are streamed into :class:`MhdNormalizer` and :class:`StokesNormalizer`, and the
    JSON files are written to ``normalization_stats/<data_source>/`` (or ``output_dir``).
    Intermediate states are saved every ``save_interval`` steps for resuming.

    Note:
        In the reframe plan the statistics are computed on the training steps only (110,
        120, 130, 140), with 4 channels and the PSF.

    Args:
        min_step: First step when scanning a range.
        max_step: Last step (inclusive) when scanning a range.
        steps: Explicit steps; overrides the range.
        save_interval: Save an intermediate state every this many steps.
        resume_from: Intermediate state to resume from.
        logtau_values: Explicit log tau nodes; override the range below.
        logtau_min: First node.
        logtau_max: Last node.
        logtau_step: Node spacing.
        use_cache: Read and write the per-step cache.
        cache_dir: Per-step cache folder.
        stokes_ic_mode: ``"per_step"`` or ``"fixed_global"`` (legacy source only).
        ic_start_step: First early step for the legacy fixed I_c.
        ic_end_step: Last early step for the legacy fixed I_c.
        ic_cont_indices: Continuum indices for the legacy fixed I_c.
        clean_start: Ignore any saved intermediate state.
        purge_cache: Delete the per-step cache first.
        data_source: ``"nicole_tau500"`` or ``"muram_legacy"``.
        data_path: The data folder.
        output_dir: Where to write the JSON files (default: the data source's folder).
        kappa_path: Rosseland opacity table (legacy source).
        lsf_path: Hinode LSF file.

    Examples:
        Statistics of the four training steps::

            python scripts/compute_normalization_stats.py --steps 110 120 130 140
    """
    # Configuration - using same paths as other scripts
    data_path = Path(data_path)
    mhd_data_dir = data_path / "muram-simulation"
    stokes_data_dir = data_path / "muram-simulation"
    kappa_path = Path(kappa_path) if kappa_path is not None else data_path / "csv/kappa.0.dat"
    lsf_path = Path(lsf_path) if lsf_path is not None else data_path / "hinode-MODEST/PSFs/hinode_sp.spline.psf"
    if output_dir is None:
        output_dir = (
            data_path / "normalization_stats"
            if data_source == "muram_legacy"
            else data_path / "normalization_stats" / data_source
        )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_reference_path = output_dir / "ic_reference_stats.json"
    raw_bz_values_dir = output_dir / "raw_bz_values"

    if ic_cont_indices is None:
        ic_cont_indices = [0, 1, 2, 3]

    valid_ic_modes = {"per_step", "fixed_global"}
    if stokes_ic_mode not in valid_ic_modes:
        raise ValueError(
            f"stokes_ic_mode must be one of {sorted(valid_ic_modes)}, got {stokes_ic_mode!r}"
        )

    if clean_start:
        print("Clean start enabled: removing previous normalization outputs...")
        for target in [
            output_dir / "mhd_normalization.json",
            output_dir / "stokes_normalization.json",
            output_dir / "intermediate_states",
            output_dir / "error_state",
            raw_bz_values_dir,
            ic_reference_path,
        ]:
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                print(f"  Removed: {target}")

        if purge_cache and use_cache:
            cache_path = Path(cache_dir)
            if cache_path.exists():
                shutil.rmtree(cache_path)
                print(f"  Removed cache directory: {cache_path}")
    
    # Simulation parameters
    nx, ny, nz = 480, 480, 256
    z_max = 250
    dz_km = 10.0

    # Use user-provided logtau nodes if passed; otherwise build from min/max/step
    if logtau_values is None or len(logtau_values) == 0:
        if logtau_step <= 0:
            raise ValueError(f"logtau_step must be > 0, got {logtau_step}")
        new_logtau = np.arange(
            logtau_min,
            logtau_max + 0.5 * logtau_step,  # robust endpoint inclusion
            logtau_step,
        ).astype(np.float32)  # accumulate in float64 first -- see TrainingConfig.get_logtau_values
    else:
        new_logtau = np.asarray(logtau_values, dtype=np.float32)

    if new_logtau.ndim != 1 or new_logtau.size < 2:
        raise ValueError("logtau_values must be 1D with at least 2 nodes.")
    if not np.all(np.diff(new_logtau) > 0):
        raise ValueError("logtau_values must be strictly increasing.")

    new_logtau = np.round(new_logtau, 6)
    n_tau = len(new_logtau)
    
    fixed_ic_value = None
    per_step_ic = None

    # Initialize normalizers
    if resume_from:
        if stokes_ic_mode == "fixed_global":
            raise ValueError(
                "resume_from is disabled for fixed_global Ic mode. "
                "Start from scratch so all stats use a single I_c reference."
            )
        print(f"Resuming from saved state: {resume_from}")
        resume_path = Path(resume_from)
        mhd_normalizer = MhdNormalizer(n_tau=n_tau, raw_values_dir=str(raw_bz_values_dir)).load_state(
            resume_path / "mhd_state.json"
        )
        stokes_normalizer = StokesNormalizer().load_state(
            resume_path / "stokes_state.json"
        )

        # Validate logtau compatibility when metadata is available
        if mhd_normalizer.logtau_values is not None:
            saved_logtau = np.asarray(mhd_normalizer.logtau_values, dtype=np.float32)
            if saved_logtau.shape != new_logtau.shape or not np.allclose(saved_logtau, new_logtau, atol=1e-6):
                raise ValueError(
                    "Resume state was computed with different logtau nodes. "
                    f"Saved: {saved_logtau.tolist()} | Requested: {new_logtau.tolist()}. "
                    "Use the same --logtau_values as the original run."
                )
        else:
            print("Warning: resume state has no logtau metadata; cannot verify compatibility.")

        # Keep metadata in-memory for future state saves
        mhd_normalizer.logtau_values = [float(x) for x in new_logtau.tolist()]

        # Extract starting step from directory name
        if "step_" in resume_path.name:
            start_step = int(resume_path.name.split("_")[1]) + 1
        else:
            start_step = min_step
        print(f"Resuming from step {start_step}")
    else:
        mhd_normalizer = MhdNormalizer(n_tau=n_tau, raw_values_dir=str(raw_bz_values_dir))
        mhd_normalizer.logtau_values = [float(x) for x in new_logtau.tolist()]
        stokes_normalizer = StokesNormalizer()
        start_step = min_step
    
    available_steps = sorted(int(s) for s in steps) if steps else list(range(start_step, max_step + 1))

    if data_source == "muram_legacy" and stokes_ic_mode == "fixed_global":
        fixed_ic_value, per_step_ic = _compute_fixed_global_ic(
            stokes_data_dir=stokes_data_dir,
            start_step=ic_start_step,
            end_step=ic_end_step,
            cont_indices=ic_cont_indices,
        )

        ic_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ic_mode": stokes_ic_mode,
            "ic_cont_indices": [int(x) for x in ic_cont_indices],
            "ic_start_step": int(ic_start_step),
            "ic_end_step": int(ic_end_step),
            "fixed_ic": float(fixed_ic_value),
            "per_step_ic": {str(k): float(v) for k, v in per_step_ic.items()},
        }
        with open(ic_reference_path, "w", encoding="utf-8") as f:
            json.dump(ic_payload, f, indent=2)
        print(f"Saved fixed I_c metadata to: {ic_reference_path}")
    elif data_source != "muram_legacy":
        print(f"data_source={data_source!r}: skipping fixed-I_c computation "
              "(source is already continuum-normalized).")

    print(f"\nComputing normalization statistics for both MHD and Stokes data")
    print(f"Processing steps {start_step} to {max_step}")
    print(f"MHD data directory: {mhd_data_dir}")
    print(f"Stokes data directory: {stokes_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Optical depth levels: {n_tau}")
    print(f"logtau nodes: {new_logtau.tolist()}")
    if logtau_values is None or len(logtau_values) == 0:
        print(f"logtau range: [{logtau_min}, {logtau_max}] step={logtau_step}")
    print(f"Use cache: {use_cache}")
    if use_cache:
        print(f"Cache dir: {cache_dir}")
    print(f"Stokes I_c mode: {stokes_ic_mode}")
    if fixed_ic_value is not None:
        print(f"Fixed global I_c: {fixed_ic_value:.6e}")
        print(f"I_c extraction steps: [{ic_start_step}, {ic_end_step}]")
        print(f"I_c continuum indices: {[int(x) for x in ic_cont_indices]}")
    print(f"Grid dimensions: ({nx}, {ny}, {nz})")
    print(f"Trimming z to: {z_max}")
    print(f"Save interval: every {save_interval} steps\n")
    
    # Build a shared training-style config to guarantee identical cache semantics
    shared_cfg = TrainingConfig(
        data_path=str(data_path),
        nx=nx, ny=ny, nz=nz, z_max=z_max, dz_km=dz_km,
        min_step=min_step, max_step=max_step,
        logtau_values=[float(x) for x in new_logtau.tolist()],
        use_cache=use_cache,
        cache_dir=cache_dir,
        stokes_cont_indices=[int(x) for x in ic_cont_indices],
        # fixed_ic policy is meaningless for pre-normalized sources; use per_step to
        # sidestep TrainingConfig's fixed_ic validation/auto-load entirely.
        stokes_ic_mode=(stokes_ic_mode if data_source == "muram_legacy" else "per_step"),
        stokes_fixed_ic=(None if fixed_ic_value is None else float(fixed_ic_value)),
        data_source=data_source,
        # avoid side effects unrelated to this script
        checkpoint_dir=str(output_dir / "_tmp_checkpoints"),
        log_dir=str(output_dir / "_tmp_logs"),
    )
    # Statistics over steps with opposite velocity signs would have a meaningless Vz mean.
    require_one_velocity_convention(available_steps, shared_cfg)

    # Initialize cache (optional)
    cache = None
    config_hash = None
    if use_cache:
        cache = MuramDataCache(cache_dir=cache_dir, compression='gzip')
        config_hash = "per-step"  # computed per step below, with the step's source fingerprint
        # strict logtau compatibility check up-front
        cache.exists(step=min_step, config_hash=None, logtau_values=new_logtau)  # validation side-effect only

    # Track progress
    successful_steps = 0
    failed_steps = []
    
    # Process each simulation step
    for step in tqdm(available_steps, desc="Processing steps"):
        try:
            if cache is not None:
                # Same per-step hash as training, so the two share cache entries and a
                # regenerated step invalidates them.
                config_hash = MuramDataCache.make_config_hash(build_cache_config_signature(shared_cfg, step=step))
            if cache is not None and config_hash is not None and cache.exists(step, config_hash, logtau_values=new_logtau):
                stokes_data_cached, mhd_data_cached, _ = cache.load_raw(step=step, verbose=False)
                stokes_data = {'I': stokes_data_cached['I'], 'V': stokes_data_cached['V']}
                mhd_data = {'T': mhd_data_cached['T'], 'Vz': mhd_data_cached['Vz'], 'Bz': mhd_data_cached['Bz']}
                print(f"\n[Step {step}] Loaded raw arrays from exact cache")
            else:
                print(f"\n[Step {step}] Processing raw {data_source} data for stats")

                stokes, mhd_od_data = load_source_arrays(step=step, config=shared_cfg)

                stokes.load_hinode_lsf(shared_cfg.data_path / shared_cfg.lsf_path)
                stokes.apply_spectral_convolution()
                stokes.resample_to_hinode()
                stokes.spectropolarimetry()

                stokes.data["mean_continuum"] = np.asarray(stokes.mean_continuum, dtype=np.float32)
                stokes.data["circular_polarization"] = np.asarray(stokes.circular_polarization, dtype=np.float32)

                stokes_data = {'I': stokes.data['I'], 'V': stokes.data['V']}
                mhd_data = {'T': mhd_od_data['T'], 'Vz': mhd_od_data['Vz'], 'Bz': mhd_od_data['Bz']}

                if cache is not None and config_hash is not None:
                    cache.save(
                        step=step,
                        stokes_data=stokes.data,
                        mhd_data=mhd_od_data,
                        approx_data={},
                        config_hash=config_hash,
                        logtau_values=new_logtau,
                        verbose=True,
                    )

            # ============ Update normalizers ============
            print(f"[Step {step}] Updating normalizers...")
            mhd_normalizer.update(mhd_data)
            stokes_normalizer.update(stokes_data)
            
            successful_steps += 1
            
            # Save intermediate state periodically
            if successful_steps % save_interval == 0:
                state_dir = output_dir / "intermediate_states" / f"step_{step:06d}"
                state_dir.mkdir(parents=True, exist_ok=True)
                
                mhd_normalizer.save_state(state_dir / "mhd_state.json")
                stokes_normalizer.save_state(state_dir / "stokes_state.json")
                
                print(f"\n[Step {step}] Saved intermediate state ({successful_steps} steps processed)")
        
        except Exception as e:
            print(f"\n[ERROR] Error processing step {step}: {e}")
            import traceback
            traceback.print_exc()
            failed_steps.append(step)
            
            # Save state before continuing
            error_dir = output_dir / "error_state" / f"step_{step:06d}"
            error_dir.mkdir(parents=True, exist_ok=True)
            mhd_normalizer.save_state(error_dir / "mhd_state.json")
            stokes_normalizer.save_state(error_dir / "stokes_state.json")
            print(f"Saved error state to {error_dir}")
            continue
    
    print("\n" + "="*60)
    print("All steps processed. Finalizing statistics...")
    print("="*60)
    print(f"  Successfully processed: {successful_steps}/{len(available_steps)} steps")
    if failed_steps:
        print(f"  Failed steps: {failed_steps}")
    print()
    
    # Finalize both normalizers
    mhd_stats = mhd_normalizer.finalize()
    stokes_stats = stokes_normalizer.finalize()
    
    # Save final statistics (include logtau metadata in MHD stats)
    mhd_normalizer.save(output_dir / "mhd_normalization.json", logtau_values=new_logtau.tolist())
    stokes_normalizer.save(output_dir / "stokes_normalization.json")
    
    print("\n" + "="*60)
    print("Normalization statistics computed successfully!")
    print("="*60)
    print(f"\nFinal files saved:")
    print(f"  MHD:    {output_dir / 'mhd_normalization.json'}")
    print(f"  Stokes: {output_dir / 'stokes_normalization.json'}")
    
    return mhd_normalizer, stokes_normalizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute normalization statistics for MHD and Stokes data"
    )
    parser.add_argument(
        "--min_step",
        type=int,
        default=60,
        help="First simulation step to process (default: 60)"
    )
    parser.add_argument(
        "--max_step",
        type=int,
        default=200,
        help="Last simulation step to process (default: 223)"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=20,
        help="Save intermediate state every N steps (default: 20)"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to directory with saved state to resume from"
    )
    parser.add_argument(
        "--logtau_values",
        type=float,
        nargs="+",
        default=None,
        help="Explicit log(tau) nodes (overrides min/max/step), e.g. --logtau_values -2.0 -1.9 ... 0.0"
    )
    parser.add_argument(
        "--logtau_min",
        type=float,
        default=-3.0,
        help="Minimum log(tau) for range mode (default: -3.0, the tau_500 generation grid)"
    )
    parser.add_argument(
        "--logtau_max",
        type=float,
        default=1.4,
        help="Maximum log(tau) for range mode (default: 1.4, the tau_500 generation grid)"
    )
    parser.add_argument(
        "--logtau_step",
        type=float,
        default=0.1,
        help="Step in log(tau) for range mode (default: 0.1)"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache usage"
    )
    parser.add_argument(
        "--stokes_ic_mode",
        type=str,
        choices=["per_step", "fixed_global"],
        default="fixed_global",
        help="Stokes continuum normalization mode (default: fixed_global)"
    )
    parser.add_argument(
        "--ic_start_step",
        type=int,
        default=70,
        help="First step (inclusive) used to compute fixed global I_c (default: 70)"
    )
    parser.add_argument(
        "--ic_end_step",
        type=int,
        default=80,
        help="Last step (inclusive) used to compute fixed global I_c (default: 80)"
    )
    parser.add_argument(
        "--ic_cont_indices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Continuum indices for I_c computation (default: 0 1 2 3)"
    )
    parser.add_argument(
        "--clean_start",
        action="store_true",
        help="Remove previous normalization outputs before running"
    )
    parser.add_argument(
        "--purge_cache",
        action="store_true",
        help="Remove cache directory before running (only used with --clean_start)"
    )
    parser.add_argument(
        "--cache-dir", "--cache_dir",
        dest="cache_dir",
        type=str,
        default=None,
        help="Cache directory (or set MURAM_CACHE_DIR). Defaults to the standard "
             ".muram_cache dir, suffixed with the data source for non-legacy sources."
    )
    parser.add_argument(
        "--data-source", "--data_source",
        dest="data_source",
        type=str,
        choices=["muram_legacy", "nicole_tau500"],
        default="nicole_tau500",
        help="Training data source (default: nicole_tau500)"
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Where to write mhd/stokes_normalization.json. Defaults to "
             "data/normalization_stats (muram_legacy) or "
             "data/normalization_stats/<data_source> otherwise."
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of steps to process (overrides --min_step/--max_step scanning); "
             "use for a small, non-contiguous set of already-generated steps."
    )

    args = parser.parse_args()

    default_cache = os.environ.get(
        "MURAM_CACHE_DIR",
        "/scratchsan/observatorio/juagudeloo/MUISCA/.muram_cache",
    )
    if args.cache_dir is None:
        args.cache_dir = (
            default_cache if args.data_source == "muram_legacy"
            else f"{default_cache}_{args.data_source}"
        )

    compute_normalization_stats(
        min_step=args.min_step,
        max_step=args.max_step,
        steps=args.steps,
        save_interval=args.save_interval,
        resume_from=args.resume_from,
        logtau_values=args.logtau_values,
        logtau_min=args.logtau_min,
        logtau_max=args.logtau_max,
        logtau_step=args.logtau_step,
        use_cache=not args.no_cache,
        cache_dir=args.cache_dir,
        data_source=args.data_source,
        output_dir=args.output_dir,
        stokes_ic_mode=args.stokes_ic_mode,
        ic_start_step=args.ic_start_step,
        ic_end_step=args.ic_end_step,
        ic_cont_indices=args.ic_cont_indices,
        clean_start=args.clean_start,
        purge_cache=args.purge_cache,
    )
