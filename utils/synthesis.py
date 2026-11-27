"""MUISCA to NICOLE forward-synthesis bridge.

A trained model's predicted atmosphere (T, v_LOS, B_LOS on its log tau nodes) is
handed to NICOLE, which synthesizes Fe I 6301.5 and 6302.5 A on the Hinode grid. The
synthetic profiles are compared with the profiles the model was given, as a check that
the predicted atmosphere reproduces the observation. The same runner also synthesizes the
MURaM training cubes (``run_cube``) and can run NICOLE inversions (``run_inversion_cube``).

* :class:`SynthesisConfig`: settings and output layout shared by the other classes.
* :class:`PredictionExporter`: runs a trained model on a region and saves the
  predicted atmospheres and the matching input profiles of chosen pixels to HDF5.
* :class:`NicoleRunner`: writes NICOLE inputs, runs NICOLE (per pixel or per cube)
  and reads the profiles back.
* :class:`SynthesisComparator`: chi-square per pixel and overlay plots.

Two sources are supported: ``"modest"`` (Hinode/SP observations) and ``"muram"`` (a
simulation step, e.g. the test step 198).

Sign conventions, checked with NICOLE (``tests/test_synthesis.py``): v_LOS > 0 shifts
the lines to the red, and B_long > 0 gives the V whose weak-field estimate is positive.
Q and U are zero unless the model carries a transverse field.

Examples:
    Synthesize the profiles of a few MODEST pixels from a trained model's prediction
    (needs the model, the MODEST files and NICOLE)::

        cfg = SynthesisConfig(source="modest", experiment_root=root, model_type="wfa_only")
        h5 = PredictionExporter(cfg, torch.device("cuda")).export([(100, 200), (50, 60)])
        NicoleRunner(cfg).run_pixels_from_h5(h5)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import astropy.units as u
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.base_training import TrainingConfig
from utils.analysis import AnalysisModelPipeline
from utils.cache_manage import ModestDataCache
from utils.hinode_wavelengths import N_WL_OBSERVED, hinode_grid_first_and_step_mA
from utils.modest_data import ModestData, transform_modest_stokes_profiles
from utils.muram_data import MhdData, StokesData
import utils.model_prof_tools as model_prof_tools
from utils.model_prof_tools import (
    check_prof,
    read_model,
    read_prof,
    write_ascii_model,
    write_binary_model_cube,
    write_prof_cube,
)
from utils.normalizer import MhdNormalizer, StokesNormalizer


def _rmtree_retry(path: Path, attempts: int = 5, delay_s: float = 0.2) -> None:
    """Delete a directory tree, retrying past NFS's transient ENOTEMPTY.

    On NFS-mounted scratch, the client's cached directory-entry count can briefly lag
    behind reality right after a subprocess closes its files, so the final ``rmdir`` inside
    ``shutil.rmtree`` occasionally raises ENOTEMPTY although every file is gone. Retrying
    after a short delay clears it; a real failure (permissions...) is printed as a warning
    after the last attempt instead of raised.

    Args:
        path: Directory to delete.
        attempts: Number of tries.
        delay_s: Pause between tries, in s.

    Examples:
        >>> import tempfile, pathlib
        >>> d = pathlib.Path(tempfile.mkdtemp()); (d / "f.txt").write_text("x")
        1
        >>> _rmtree_retry(d); d.exists()
        False
    """
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                print(f"  WARNING: failed to remove workdir {path}: {exc}")
                return
            time.sleep(delay_s)


@dataclass
class WholeRegionPrediction:
    """A model's prediction over a whole region, with the profiles it was given.

    Attributes:
        pred_mhd: ``{"T", "Vz", "Bz"}`` in K, km/s and G, each ``(pred_nx, pred_ny, n_tau)``.
        prediction_stokes: The model's input ``{"I", "V"}`` before statistical
            normalization (continuum units), each ``(pred_nx, pred_ny, n_wl)``.
        wavelength: Wavelength axis in Angstrom.
        logtau: The model's log tau nodes.
        pred_nx: Size of the prediction grid along the first axis.
        pred_ny: Size along the second axis.
        gt_pressure: MURaM gas pressure on the model's nodes (dyn cm^-2), only with
            ``add_gt_pressure``; otherwise ``None``.
    """
    pred_mhd: dict[str, np.ndarray]          # {"T", "Vz", "Bz"}, each (pred_nx, pred_ny, n_tau)
    prediction_stokes: dict[str, np.ndarray] # {"I", "V"}, each (pred_nx, pred_ny, n_wl)
    wavelength: np.ndarray                    # (n_wl,)
    logtau: np.ndarray                        # (n_tau,)
    pred_nx: int
    pred_ny: int
    # Ground-truth gas pressure (muram source, --add-gt-pressure only), remapped
    # onto the SAME grid as `logtau` (the model's own predicted tau grid) so it
    # can sit alongside T/Vz/Bz in one NICOLE .model file per pixel. dyn/cm^2.
    gt_pressure: Optional[np.ndarray] = None  # (pred_nx, pred_ny, n_tau) or None


@dataclass
class SynthesisConfig:
    """Settings of a synthesis run and the layout of its output folders.

    The NICOLE wavelength grid defaults to the Hinode grid read from the MODEST FITS
    header (``utils.hinode_wavelengths``). It used to be hardcoded at 6300.796 A with
    21.5 mA steps, 0.0776 A (3.7 km/s) off the observed axis.

    Output layout: ``output_root/experiment_root/modest/<model_type>/<region_label>/`` for
    MODEST and ``output_root/experiment_root/muram/step-<N>[-gt-pressure]/<model_type>/``
    for MURaM. Each holds ``predictions.h5`` and ``syntheses.h5``.

    Attributes:
        source: ``"modest"`` or ``"muram"``.
        experiment_root: Experiment folder name under ``output/experiments``.
        model_type: Flavor, e.g. ``"wfa_only"``.
        region_label: MODEST region name (``"whole"``, ``"sunspot"``...).
        crop_bounds: MODEST crop ``(y0, y1, x0, x1)`` on the observed grid.
        muram_step: MURaM step for ``source="muram"``.
        add_gt_pressure: Give NICOLE MURaM's gas pressure instead of solving
            hydrostatic equilibrium from a flat electron-pressure seed.
        stokes_v_multiplier: Factor on observed V; -1 by default (see
            :class:`utils.modest_data.ModestData`).
        prediction_input_mode: MODEST input grid, ``"upsampled"`` or ``"downsampled"``.
        wl_first: First wavelength in A (default: from the FITS header).
        wl_step_mA: Wavelength step in mA (default: from the FITS header).
        n_wl: Number of wavelengths (112).
        v_mic_cms: Microturbulence given to NICOLE, in cm/s (MUISCA does not predict it).
        v_mac_cms: Macroturbulence in cm/s.
        stray_light: Stray-light fraction.
        el_p_seed: Flat electron-pressure seed (dyn cm^-2) for hydrostatic equilibrium.
        python_for_nicole: Interpreter for NICOLE v16.06's Python 2 driver.
        inference_batch_size: Batch size for the model.
    """
    source: str  # "modest" or "muram"
    experiment_root: str
    model_type: str
    region_label: str = "whole"  # modest only; ignored when source == "muram"
    crop_bounds: Optional[tuple[int, int, int, int]] = None  # passed verbatim to ModestData; modest only

    # MURaM loader (ignored when source == "modest")
    muram_step: Optional[int] = None
    add_gt_pressure: bool = False  # feed NICOLE the true MURaM gas pressure instead of a hydrostatic seed

    # Filesystem layout
    output_root: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/output/synthesis")
    nicole_root: Path = Path("/scratchsan/observatorio/juagudeloo/NICOLE_v16.06")
    nicole_assets: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/data/nicole_assets")

    # MODEST loader
    modest_cache_dir: Path = Path("/scratchsan/observatorio/juagudeloo/MUISCA/.modest_cache")
    use_modest_cache: bool = True
    apply_polarization_mask: bool = False
    polarization_threshold: float = 1e-2
    stokes_v_multiplier: float = -1.0
    prediction_input_mode: str = "upsampled"

    # NICOLE wavelength grid (MODEST/Hinode SP Fe I 6301.5/6302.5 window).
    # Left as None so they are derived from the MODEST FITS header in
    # __post_init__ (utils.hinode_wavelengths) rather than hardcoded: these
    # were 6300.796 / 21.5, which disagreed with the observed axis by 0.0776 A
    # (~3.7 km/s) and made profiles synthesized from MUISCA predictions
    # spectrally misaligned with the observations they were compared against.
    # Set explicitly only to override.
    wl_first: Optional[float] = None
    wl_step_mA: Optional[float] = None
    n_wl: int = N_WL_OBSERVED

    # NICOLE atmospheric-column defaults (for the values MUISCA does not predict)
    v_mic_cms: float = 1.0e5
    v_mac_cms: float = 0.0
    stray_light: float = 0.0
    el_p_seed: float = 1.0

    # NICOLE driver interpreter (v16.06's run_nicole.py is Python 2)
    python_for_nicole: str = "python2"

    # Inference
    inference_batch_size: int = 4096

    def __post_init__(self) -> None:
        # Derive the spectral grid from the MODEST FITS header unless explicitly
        # overridden, so synthesis always lands on the same axis the observations
        # are loaded on (see utils.hinode_wavelengths).
        """Fill the wavelength grid from the MODEST FITS header unless it was given."""
        if self.wl_first is None or self.wl_step_mA is None:
            first, step_mA = hinode_grid_first_and_step_mA(self.n_wl)
            if self.wl_first is None:
                self.wl_first = first
            if self.wl_step_mA is None:
                self.wl_step_mA = step_mA

    def _source_root(self) -> Path:
        """Folder of this source: ``.../modest`` or ``.../muram/step-<N>[-gt-pressure]``."""
        root = self.output_root / self.experiment_root
        if self.source == "muram":
            step_label = f"step-{self.muram_step}"
            if self.add_gt_pressure:
                step_label += "-gt-pressure"
            root = root / "muram" / step_label
        else:
            root = root / "modest"
        return root

    def out_dir(self) -> Path:
        """Folder of this model's predictions and syntheses.

        Returns:
            ``<source root>/<model_type>/<region_label>`` (MODEST) or
            ``<source root>/<model_type>`` (MURaM).

        Examples:
            >>> cfg = SynthesisConfig("muram", "exp", "wfa_only", muram_step=198, output_root=Path("/o"),
            ...                       wl_first=6300.8736, wl_step_mA=21.477)
            >>> str(cfg.out_dir())
            '/o/exp/muram/step-198/wfa_only'
        """
        root = self._source_root()
        if self.source == "muram":
            return root / self.model_type  # no region_label segment: step-N plays that role
        return root / self.model_type / self.region_label

    def region_dir(self) -> Path:
        """Folder shared by all model flavors (where cross-model comparisons are written).

        Returns:
            ``<source root>/<region_label>`` (MODEST) or ``<source root>`` (MURaM).

        Examples:
            >>> cfg = SynthesisConfig("modest", "exp", "wfa_only", region_label="sunspot", output_root=Path("/o"),
            ...                       wl_first=6300.8736, wl_step_mA=21.477)
            >>> str(cfg.region_dir())
            '/o/exp/modest/sunspot'
        """
        root = self._source_root()
        if self.source == "muram":
            return root  # step-N *is* the shared level
        return root / self.region_label

    def sampling_dir(self) -> Path:
        """Folder whose ``pixel_selection/`` holds the selected pixels.

        The pixel selection does not depend on the gas pressure, so a run with
        ``add_gt_pressure`` and its plain sibling must use the same pixels; both read them from
        the plain run's folder.

        Returns:
            :meth:`out_dir` of the same configuration without ``add_gt_pressure``.

        Examples:
            >>> cfg = SynthesisConfig("muram", "exp", "base", muram_step=198, add_gt_pressure=True,
            ...                       output_root=Path("/o"), wl_first=6300.8736, wl_step_mA=21.477)
            >>> str(cfg.out_dir()), str(cfg.sampling_dir())
            ('/o/exp/muram/step-198-gt-pressure/base', '/o/exp/muram/step-198/base')
        """
        if self.source == "muram" and self.add_gt_pressure:
            import dataclasses
            return dataclasses.replace(self, add_gt_pressure=False).out_dir()
        return self.out_dir()

    def predictions_h5(self) -> Path:
        """Path of ``predictions.h5`` in :meth:`out_dir`."""
        return self.out_dir() / "predictions.h5"

    def syntheses_h5(self) -> Path:
        """Path of ``syntheses.h5`` in :meth:`out_dir`."""
        return self.out_dir() / "syntheses.h5"


class PredictionExporter:
    """Run a trained model on a region and save the atmospheres and input profiles of chosen pixels."""

    def __init__(self, cfg: SynthesisConfig, device: torch.device):
        """Check the configuration.

        Args:
            cfg: Synthesis configuration.
            device: Torch device for inference.

        Raises:
            NotImplementedError: For an unknown source.
            ValueError: If ``muram_step`` is missing for MURaM, or ``add_gt_pressure`` is set
                for MODEST.
        """
        if cfg.source not in ("modest", "muram"):
            raise NotImplementedError(
                f"PredictionExporter supports source in ('modest', 'muram') "
                f"(got {cfg.source!r})"
            )
        if cfg.source == "muram" and cfg.muram_step is None:
            raise ValueError("cfg.muram_step must be set when source='muram'")
        if cfg.add_gt_pressure and cfg.source != "muram":
            raise ValueError("add_gt_pressure is only supported for source='muram'")
        self.cfg = cfg
        self.device = device

    def predict_whole_region(self) -> WholeRegionPrediction:
        """Run the model over the whole region and return physical-unit atmospheres.

        Loads the checkpoint of ``cfg.model_type``, the normalization statistics of
        ``TrainingConfig()`` and the region's profiles, normalizes them, predicts and
        denormalizes.

        Returns:
            A :class:`WholeRegionPrediction`.

        Raises:
            RuntimeError: If no model or no log tau grid can be found for the experiment.

        Examples:
            >>> region = PredictionExporter(cfg, torch.device("cuda")).predict_whole_region()  # doctest: +SKIP
            >>> region.pred_mhd["T"].shape  # doctest: +SKIP
            (780, 1716, 45)
        """
        cfg = self.cfg
        out_dir = cfg.out_dir()
        out_dir.mkdir(parents=True, exist_ok=True)

        pipeline = AnalysisModelPipeline(
            device=self.device,
            output_dir=out_dir,
            experiment_root=cfg.experiment_root,
        )
        model_configs, models, n_tau = pipeline.prepare_models([cfg.model_type])
        if not models:
            raise RuntimeError(
                f"No model loaded for experiment_root={cfg.experiment_root!r}, "
                f"model_type={cfg.model_type!r}"
            )
        model_name, model = next(iter(models.items()))
        model_cfg = model_configs[model_name]
        pred_tau = pipeline.get_model_logtau_values(model_cfg)
        if pred_tau is None:
            raise RuntimeError(
                f"Could not determine logtau values for model {model_name}"
            )
        pred_tau_arr = np.asarray(pred_tau, dtype=np.float64)

        # Normalizers (loaded once)
        default_cfg = TrainingConfig()
        mhd_normalizer = MhdNormalizer()
        stokes_normalizer = StokesNormalizer()
        mhd_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.mhd_normalizer_path))
        stokes_normalizer.load(filepath=str(Path(default_cfg.data_path) / default_cfg.stokes_normalizer_path))

        gt_pressure = None
        if cfg.source == "modest":
            prediction_stokes, wavelength = self._load_modest_stokes(cfg)
        else:  # cfg.source == "muram"
            prediction_stokes, wavelength, gt_pressure = self._load_muram_stokes(
                cfg, pipeline, model_cfg, pred_tau_arr
            )

        pred_nx, pred_ny = prediction_stokes["I"].shape[:2]
        norm_stokes = stokes_normalizer.transform(prediction_stokes)
        I_flat = norm_stokes["I"].reshape(pred_nx * pred_ny, -1)
        V_flat = norm_stokes["V"].reshape(pred_nx * pred_ny, -1)
        stokes_input = np.stack([I_flat, V_flat], axis=1).astype(np.float32)

        # Inference
        pred_mhd = pipeline.predict_and_denormalize(
            model=model,
            stokes_input=stokes_input,
            mhd_normalizer=mhd_normalizer,
            pred_nx=pred_nx,
            pred_ny=pred_ny,
            batch_size=cfg.inference_batch_size,
        )

        return WholeRegionPrediction(
            pred_mhd=pred_mhd,
            prediction_stokes=prediction_stokes,
            wavelength=wavelength,
            logtau=pred_tau_arr,
            pred_nx=pred_nx,
            pred_ny=pred_ny,
            gt_pressure=gt_pressure,
        )

    @staticmethod
    def _load_modest_stokes(cfg: SynthesisConfig) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """MODEST profiles prepared as in the analysis (``ModestData.load_all``), and their wavelength axis.

        Returns the continuum-normalized (not statistically normalized) ``{"I", "V"}`` cubes.
        """
        modest = ModestData(
            circular_polarization_threshold=cfg.polarization_threshold,
            stokes_v_multiplier=cfg.stokes_v_multiplier,
        )
        modest_cache = ModestDataCache(cache_dir=cfg.modest_cache_dir)
        modest_data = modest.load_all(
            region_bounds=tuple(cfg.crop_bounds) if cfg.crop_bounds is not None else None,
            apply_mask=cfg.apply_polarization_mask,
            cache=modest_cache,
            use_cache=cfg.use_modest_cache,
            prediction_input_mode=cfg.prediction_input_mode,
        )

        prediction_stokes = modest_data.get("prediction_stokes", modest_data["smoothed_stokes"])
        wl_raw = modest_data.get("wl", np.arange(prediction_stokes["I"].shape[-1]))
        wavelength = np.asarray(
            wl_raw.value if hasattr(wl_raw, "value") else wl_raw,
            dtype=np.float64,
        )
        # No spectral transforms in v1 -- defaults match the analysis script's
        # no-shift, no-scale, no-invert path.
        prediction_stokes = transform_modest_stokes_profiles(
            stokes=prediction_stokes,
            wavelength_angstrom=wavelength,
            shift_positions=0.0,
            i_scale=1.0,
            v_scale=1.0,
            invert_direction=False,
        )
        return prediction_stokes, wavelength

    @staticmethod
    def _load_muram_stokes(
        cfg: SynthesisConfig,
        pipeline: AnalysisModelPipeline,
        model_cfg: dict,
        pred_tau_arr: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """MURaM profiles of ``cfg.muram_step`` prepared as at training time, and optionally the gas pressure.

        Note:
            This reads the legacy ``stokes_<step>.npy`` cube (continuum normalization of the
            old pipeline, Rosseland tau for the pressure) regardless of the checkpoint's data
            source. For models trained on ``nicole_tau500`` the MURaM input therefore differs
            from the training profiles (e.g. continuum 0.91 instead of 0.97 at one pixel of
            step 198). Fixed in the reframe plan, Phase 1d. Until then it refuses models
            trained with the spatial PSF (``NotImplementedError``), and runs older than Phase
            1b are refused by ``build_runtime_training_config``, so it only serves runs
            trained with ``--spatial-psf 0``.

        Returns ``(prediction_stokes, wavelength, gt_pressure)``.
        """
        # Reuse the exact per-model config (LSF path, continuum-normalization
        # mode, stokes_mult_factor, ...) the checkpoint was actually trained
        # with -- same call scripts/analysis/muram_analysis.py makes.
        train_cfg = pipeline.build_runtime_training_config(model_cfg)
        if train_cfg.spatial_psf:
            # This path does not apply the spatial PSF, so a PSF-trained model would get
            # profiles with MURaM's full granulation contrast. Phase 1d moves it onto
            # load_source_arrays + degrade_training_stokes.
            raise NotImplementedError(
                "The MURaM bridge does not degrade the profiles with the spatial PSF yet (reframe "
                "Phase 1d); it only serves models trained without it."
            )
        data_path = Path(train_cfg.data_path)

        mhd = MhdData(
            data_path=data_path / "muram-simulation",
            nx=train_cfg.nx,
            ny=train_cfg.ny,
            nz=train_cfg.nz,
        )
        mhd.load_step(step=cfg.muram_step, z_max=train_cfg.z_max)

        stokes = StokesData(
            data_dir=data_path / "muram-simulation",
            step=cfg.muram_step,
            wavelength_range=(6300.5, 6303.5),
            wavelength_step=0.01,
        )
        stokes.load_stokes()
        stokes_cont_indices = train_cfg.stokes_cont_indices or [0, 1, 2, 3]
        fixed_ic = (
            float(train_cfg.stokes_fixed_ic) if train_cfg.stokes_ic_mode == "fixed_global" else None
        )
        stokes.continuum_normalization(cont_indices=stokes_cont_indices, fixed_ic=fixed_ic)
        if train_cfg.stokes_mult_factor != 1.0:
            stokes.data["I"] = stokes.data["I"] * train_cfg.stokes_mult_factor
            stokes.data["V"] = stokes.data["V"] * train_cfg.stokes_mult_factor
        stokes.load_hinode_lsf(data_path / train_cfg.lsf_path)
        stokes.apply_spectral_convolution()
        stokes.resample_to_hinode()

        prediction_stokes = {"I": stokes.data["I"], "V": stokes.data["V"]}
        wavelength = np.asarray(stokes.hinode_wl, dtype=np.float64)

        gt_pressure = None
        if cfg.add_gt_pressure:
            mhd.load_opacity_table(kappa_path=data_path / train_cfg.kappa_path)
            mhd.compute_optical_depth(dz=train_cfg.dz_km * u.km)
            mhd.remap_to_optical_depth(pred_tau_arr, quantities=["P"])
            p_arr = mhd.od_data["P"]
            gt_pressure = np.asarray(
                p_arr.value if hasattr(p_arr, "value") else p_arr, dtype=np.float64
            )

        return prediction_stokes, wavelength, gt_pressure

    def export(self, pixels: Sequence[tuple[int, int]]) -> Path:
        """Predict the whole region and save the chosen pixels to ``predictions.h5``.

        The file holds ``T`` (K), ``Vz`` (km/s) and ``Bz`` (G) as ``(n, n_tau)``, ``pixels``
        ``(n, 2)``, ``logtau``, ``wavelengths`` (A), ``stokes_obs`` ``(n, 2, n_wl)`` with the
        model's input I and V, and ``Pgas_gt`` when requested.

        Args:
            pixels: ``(ix, iy)`` on the prediction grid (for MODEST, ix is the row).

        Returns:
            Path of ``predictions.h5``.

        Raises:
            IndexError: If a pixel is outside the prediction grid.

        Examples:
            >>> h5 = PredictionExporter(cfg, torch.device("cuda")).export([(100, 200)])  # doctest: +SKIP
        """
        cfg = self.cfg

        # Run inference over the whole region once
        region = self.predict_whole_region()
        pred_mhd = region.pred_mhd
        prediction_stokes = region.prediction_stokes
        wavelength = region.wavelength
        logtau = region.logtau
        pred_nx = region.pred_nx
        pred_ny = region.pred_ny
        n_tau = len(logtau)

        # Subselect the requested pixels.
        pixels_arr = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
        for (ix, iy) in pixels_arr:
            if not (0 <= ix < pred_nx and 0 <= iy < pred_ny):
                raise IndexError(
                    f"Pixel (ix={ix}, iy={iy}) out of range for prediction grid "
                    f"{pred_nx}x{pred_ny}"
                )

        n = pixels_arr.shape[0]
        T_out = np.empty((n, n_tau), dtype=np.float64)
        Vz_out = np.empty((n, n_tau), dtype=np.float64)
        Bz_out = np.empty((n, n_tau), dtype=np.float64)
        stokes_obs = np.empty((n, 2, prediction_stokes["I"].shape[-1]), dtype=np.float32)
        Pgas_gt_out = np.empty((n, n_tau), dtype=np.float64) if region.gt_pressure is not None else None
        for k, (ix, iy) in enumerate(pixels_arr):
            T_out[k] = pred_mhd["T"][ix, iy, :n_tau]
            Vz_out[k] = pred_mhd["Vz"][ix, iy, :n_tau]
            Bz_out[k] = pred_mhd["Bz"][ix, iy, :n_tau]
            stokes_obs[k, 0] = prediction_stokes["I"][ix, iy, :]
            stokes_obs[k, 1] = prediction_stokes["V"][ix, iy, :]
            if Pgas_gt_out is not None:
                Pgas_gt_out[k] = region.gt_pressure[ix, iy, :n_tau]

        out_path = cfg.predictions_h5()
        if out_path.exists():
            out_path.unlink()
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset("T", data=T_out)
            h5.create_dataset("Vz", data=Vz_out)
            h5.create_dataset("Bz", data=Bz_out)
            h5.create_dataset("pixels", data=pixels_arr)
            h5.create_dataset("logtau", data=logtau)
            h5.create_dataset("wavelengths", data=wavelength)
            h5.create_dataset("stokes_obs", data=stokes_obs)
            if Pgas_gt_out is not None:
                h5.create_dataset("Pgas_gt", data=Pgas_gt_out)
            h5.attrs["source"] = cfg.source
            h5.attrs["experiment_root"] = cfg.experiment_root
            h5.attrs["model_type"] = cfg.model_type
            h5.attrs["region_label"] = cfg.region_label
            h5.attrs["pred_grid"] = np.array([pred_nx, pred_ny], dtype=np.int64)
            h5.attrs["output_root"] = str(cfg.output_root)
            if cfg.crop_bounds is not None:
                h5.attrs["crop_bounds"] = np.asarray(cfg.crop_bounds, dtype=np.int64)
            if cfg.source == "muram":
                h5.attrs["muram_step"] = int(cfg.muram_step)
                h5.attrs["add_gt_pressure"] = bool(cfg.add_gt_pressure)
        return out_path


class NicoleRunner:
    """Write NICOLE inputs, run NICOLE and read its profiles back.

    NICOLE v16.06 is driven through its Python 2 ``run_nicole.py``; the Fortran binary
    must be built in ``nicole_root/main``. NICOLE's v_LOS is positive for a redshift and its
    B_long has the sign of the WFA estimate from the synthesized V.
    """

    def __init__(self, cfg: SynthesisConfig):
        """Find the NICOLE templates, line list, binary and driver.

        Args:
            cfg: Synthesis configuration (``nicole_root`` and ``nicole_assets`` are used).

        Raises:
            FileNotFoundError: If a template, the line list, the binary or the driver is
                missing.
        """
        self.cfg = cfg
        self.template_path = cfg.nicole_assets / "NICOLE.input.template"
        self.inversion_template_path = cfg.nicole_assets / "NICOLE.input_inversion.template"
        self.lines_path = cfg.nicole_assets / "LINES"
        if not self.template_path.exists():
            raise FileNotFoundError(self.template_path)
        if not self.lines_path.exists():
            raise FileNotFoundError(self.lines_path)
        nicole_command = cfg.nicole_root / "main" / "nicole"
        if not nicole_command.exists():
            raise FileNotFoundError(
                f"NICOLE binary not found at {nicole_command}. "
                "Build NICOLE first (cd NICOLE/main && make)."
            )
        self.nicole_command = nicole_command
        self.run_nicole_py = cfg.nicole_root / "run" / "run_nicole.py"
        if not self.run_nicole_py.exists():
            raise FileNotFoundError(self.run_nicole_py)

    def _pixel_workdir(self, ix: int, iy: int) -> Path:
        """Working folder of one pixel, ``pix_<ix>_<iy>`` in :meth:`SynthesisConfig.out_dir`."""
        return self.cfg.out_dir() / f"pix_{ix:05d}_{iy:05d}"

    def prepare_workdir(
        self,
        ix: int,
        iy: int,
        logtau: np.ndarray,
        T: np.ndarray,
        v_los_kms: np.ndarray,
        b_long_G: np.ndarray,
        gt_pressure: Optional[np.ndarray] = None,
    ) -> Path:
        """Write one pixel's ASCII model, ``NICOLE.input``, ``LINES`` and the driver link.

        Without ``gt_pressure`` NICOLE solves hydrostatic equilibrium from a flat
        electron-pressure seed; with it, the gas pressure is used as given.

        Args:
            ix: Pixel index along the first axis.
            iy: Pixel index along the second axis.
            logtau: Nodes, increasing.
            T: Temperature per node, K.
            v_los_kms: Line-of-sight velocity per node, km/s (positive = redshift).
            b_long_G: Longitudinal field per node, G.
            gt_pressure: Gas pressure per node, dyn cm^-2, or ``None``.

        Returns:
            The working folder.

        Examples:
            >>> wd = runner.prepare_workdir(100, 200, logtau, T, v, b)  # doctest: +SKIP
        """
        workdir = self._pixel_workdir(ix, iy)
        workdir.mkdir(parents=True, exist_ok=True)

        # Write the per-pixel atmosphere. When gt_pressure is supplied (true
        # MURaM gas pressure, --add-gt-pressure), use it verbatim instead of a
        # flat hydrostatic seed, and tell NICOLE not to override it.
        model_path = workdir / "model.model"
        if gt_pressure is not None:
            write_ascii_model(
                model_path,
                logtau=logtau,
                T=T,
                v_los_kms=v_los_kms,
                b_long_G=b_long_G,
                v_mic_cms=self.cfg.v_mic_cms,
                v_mac_cms=self.cfg.v_mac_cms,
                stray_light=self.cfg.stray_light,
                el_p=gt_pressure,
            )
            hydrostatic_eq, input_density = "N", "PGas"
        else:
            write_ascii_model(
                model_path,
                logtau=logtau,
                T=T,
                v_los_kms=v_los_kms,
                b_long_G=b_long_G,
                v_mic_cms=self.cfg.v_mic_cms,
                v_mac_cms=self.cfg.v_mac_cms,
                stray_light=self.cfg.stray_light,
                el_p_seed=self.cfg.el_p_seed,
            )
            hydrostatic_eq, input_density = "Y", "Pel"

        # Substitute placeholders in NICOLE.input template
        template = self.template_path.read_text()
        rendered = template.format(
            NICOLE_COMMAND=str(self.nicole_command),
            INPUT_MODEL="model.model",
            OUTPUT_PROFILE="profile.pro",
            OUTPUT_MODEL="model_out.mod",
            WL_FIRST=f"{self.cfg.wl_first:.4f}",
            WL_STEP_MA=f"{self.cfg.wl_step_mA:g}",
            N_WL=int(self.cfg.n_wl),
            HYDROSTATIC_EQ=hydrostatic_eq,
            INPUT_DENSITY=input_density,
        )
        (workdir / "NICOLE.input").write_text(rendered)

        # Drop the LINES file in place (copy keeps the workdir self-contained).
        shutil.copyfile(self.lines_path, workdir / "LINES")

        # NICOLE's run_nicole.py validates its own source via a relative
        # open('run_nicole.py'); the canonical pattern (cf. test/syn1) is a
        # symlink into the working directory. Replace any stale link.
        link = workdir / "run_nicole.py"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.run_nicole_py)

        return workdir

    def run_pixel(self, workdir: Path) -> np.ndarray:
        """Run NICOLE in a prepared working folder and return the ``(4, n_wl)`` I, Q, U, V profile.

        Args:
            workdir: Folder from :meth:`prepare_workdir`.

        Returns:
            The profile in continuum units.

        Raises:
            RuntimeError: If NICOLE exits with an error.
            FileNotFoundError: If NICOLE writes no profile.

        Examples:
            >>> iquv = runner.run_pixel(wd)  # doctest: +SKIP
        """
        env = os.environ.copy()
        # v16.06's run_nicole.py only initializes inputmodel/outprof when --modelin
        # and --profout are passed explicitly; without them, NameError on line 184.
        result = subprocess.run(
            [
                self.cfg.python_for_nicole,
                "./run_nicole.py",
                "--nicolecommand=" + str(self.nicole_command),
                "--modelin=model.model",
                "--profout=profile.pro",
            ],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NICOLE failed in {workdir} (exit {result.returncode}).\n"
                f"--- STDOUT ---\n{result.stdout}\n"
                f"--- STDERR ---\n{result.stderr}"
            )
        prof_path = workdir / "profile.pro"
        if not prof_path.exists():
            raise FileNotFoundError(
                f"NICOLE did not produce expected output {prof_path}.\n"
                f"--- STDOUT ---\n{result.stdout}"
            )
        # v16.06 writes 'nicole2.3bp' binary profiles. The shipped check_prof
        # opens in text mode and fails to decode under Python 3, but we already
        # know the format and shape from the NICOLE.input region we rendered.
        flat = read_prof(
            str(prof_path), "nicole2.3", 1, 1, int(self.cfg.n_wl), 0, 0
        )
        # read_prof's binary-format branch opens the file as a module-level
        # global ('f') and never closes it (designed for sequential multi-record
        # reads of one file, which isn't our usage -- one fresh file per pixel).
        # Close it explicitly so the workdir cleanup below doesn't hit an NFS
        # "device busy" silly-rename on a still-open profile.pro.
        f_handle = getattr(model_prof_tools, "f", None)
        if f_handle is not None and not f_handle.closed:
            f_handle.close()
        flat = np.asarray(flat, dtype=np.float64).reshape(int(self.cfg.n_wl), 4)
        # NICOLE row layout per wavelength: [I, Q, U, V]
        iquv = np.stack([flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]], axis=0)
        return iquv

    def run_cube(
        self,
        workdir: Path,
        logtau: np.ndarray,
        T: np.ndarray,
        v_los_kms: np.ndarray,
        b_long_G: np.ndarray,
        n_wl: int,
        *,
        gt_pressure: Optional[np.ndarray] = None,
        wl_first: Optional[float] = None,
        wl_step_mA: Optional[float] = None,
        v_mic_cms: Optional[float] = None,
        b_x_G: Optional[np.ndarray] = None,
        b_y_G: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Synthesize a whole ``(nx, ny)`` atmosphere cube in one NICOLE run.

        Writes the cube as a binary model (``write_binary_model_cube``), renders
        ``NICOLE.input`` for the wavelength grid, runs NICOLE once (its Fortran loops over the
        pixels) and reads the profiles back.

        A warm-up pixel is prepended and discarded. NICOLE gives a slightly different result
        (about 3e-3) for a pixel whose predecessor is identical or absent, so the warm-up is a
        copy of the first pixel with T times 1.05; every real pixel then has a different
        predecessor, and the result does not depend on how a frame is split into chunks.

        Without ``gt_pressure`` the flat electron-pressure seed and hydrostatic equilibrium are
        used (as for the training data); with it, the gas pressure is used as given.

        Args:
            workdir: Folder for NICOLE's files.
            logtau: Nodes, increasing, shared by all pixels.
            T: ``(nx, ny, nz)`` temperature, K.
            v_los_kms: ``(nx, ny, nz)`` line-of-sight velocity, km/s (positive = redshift).
            b_long_G: ``(nx, ny, nz)`` longitudinal field, G.
            n_wl: Number of wavelengths.
            gt_pressure: ``(nx, ny, nz)`` gas pressure, dyn cm^-2, or ``None``.
            wl_first: First wavelength in A (default: the configuration's).
            wl_step_mA: Step in mA (default: the configuration's).
            v_mic_cms: Microturbulence in cm/s (default: the configuration's).
            b_x_G: ``(nx, ny, nz)`` transverse field along x in G, or ``None`` (zero).
            b_y_G: ``(nx, ny, nz)`` transverse field along y in G, or ``None`` (zero).
                Q and U are zero without a transverse field.

        Returns:
            ``(nx, ny, 4, n_wl)`` I, Q, U, V in continuum units.

        Raises:
            RuntimeError: If NICOLE exits with an error.
            FileNotFoundError: If NICOLE writes no profile.

        Examples:
            >>> iquv = runner.run_cube(workdir, logtau, T, v, b, 112)  # doctest: +SKIP
            >>> iquv.shape  # doctest: +SKIP
            (2, 3, 4, 112)
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        nx, ny = T.shape[:2]
        nz = T.shape[2]

        wl_first = self.cfg.wl_first if wl_first is None else wl_first
        wl_step_mA = self.cfg.wl_step_mA if wl_step_mA is None else wl_step_mA
        v_mic_cms = self.cfg.v_mic_cms if v_mic_cms is None else v_mic_cms

        # Flatten to (npix, nz) and prepend a warm-up pixel laid out as a
        # 1 x (npix+1) cube. The warm-up is a DISTINCT copy of pixel 0 (T
        # scaled by 1.05) so that pixel 0 -- like every other real pixel --
        # is preceded by a different atmosphere (see docstring).
        npix = nx * ny
        Tf = T.reshape(npix, nz)
        Vf = v_los_kms.reshape(npix, nz)
        Bf = b_long_G.reshape(npix, nz)
        Tw = np.concatenate([Tf[:1] * 1.05, Tf], axis=0).reshape(1, npix + 1, nz)
        Vw = np.concatenate([Vf[:1], Vf], axis=0).reshape(1, npix + 1, nz)
        Bw = np.concatenate([Bf[:1], Bf], axis=0).reshape(1, npix + 1, nz)
        Pw = None
        if gt_pressure is not None:
            Pf = gt_pressure.reshape(npix, nz)
            Pw = np.concatenate([Pf[:1], Pf], axis=0).reshape(1, npix + 1, nz)
        transverse = {}
        for key, arr in (("b_x_G", b_x_G), ("b_y_G", b_y_G)):
            if arr is not None:
                if np.shape(arr) != T.shape:
                    raise ValueError(f"{key} has shape {np.shape(arr)}; it must match T, {T.shape}")
                af = np.asarray(arr, dtype=np.float64).reshape(npix, nz)
                transverse[key] = np.concatenate([af[:1], af], axis=0).reshape(1, npix + 1, nz)

        model_path = workdir / "model.model"
        if gt_pressure is not None:
            write_binary_model_cube(
                model_path,
                logtau=logtau, T=Tw, v_los_kms=Vw, b_long_G=Bw,
                v_mic_cms=v_mic_cms, v_mac_cms=self.cfg.v_mac_cms,
                stray_light=self.cfg.stray_light, gas_p=Pw, **transverse,
            )
            hydrostatic_eq, input_density = "N", "PGas"
        else:
            write_binary_model_cube(
                model_path,
                logtau=logtau, T=Tw, v_los_kms=Vw, b_long_G=Bw,
                v_mic_cms=v_mic_cms, v_mac_cms=self.cfg.v_mac_cms,
                stray_light=self.cfg.stray_light, el_p_seed=self.cfg.el_p_seed, **transverse,
            )
            hydrostatic_eq, input_density = "Y", "Pel"

        template = self.template_path.read_text()
        rendered = template.format(
            NICOLE_COMMAND=str(self.nicole_command),
            INPUT_MODEL="model.model",
            OUTPUT_PROFILE="profile.pro",
            OUTPUT_MODEL="model_out.mod",
            WL_FIRST=f"{wl_first:.4f}",
            WL_STEP_MA=f"{wl_step_mA:g}",
            N_WL=int(n_wl),
            HYDROSTATIC_EQ=hydrostatic_eq,
            INPUT_DENSITY=input_density,
        )
        (workdir / "NICOLE.input").write_text(rendered)
        shutil.copyfile(self.lines_path, workdir / "LINES")
        link = workdir / "run_nicole.py"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.run_nicole_py)

        env = os.environ.copy()
        result = subprocess.run(
            [
                self.cfg.python_for_nicole,
                "./run_nicole.py",
                "--nicolecommand=" + str(self.nicole_command),
                "--modelin=model.model",
                "--profout=profile.pro",
            ],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NICOLE cube run failed in {workdir} (exit {result.returncode}).\n"
                f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
            )
        prof_path = workdir / "profile.pro"
        if not prof_path.exists():
            raise FileNotFoundError(
                f"NICOLE did not produce {prof_path}.\n--- STDOUT ---\n{result.stdout}"
            )

        # Profile cube is 1 x (npix+1): record 0 is the warm-up pixel (discarded),
        # records 1..npix are the real pixels in flattened (ix*ny + iy) order.
        # read_prof returns 4*n_wl interleaved [I,Q,U,V]; read sequentially.
        prof_ny = npix + 1
        flat_out = np.empty((npix, 4, int(n_wl)), dtype=np.float64)
        seq = 0
        for p in range(prof_ny):
            flat = read_prof(
                str(prof_path), "nicole2.3", 1, prof_ny, int(n_wl), 0, p, sequential=seq
            )
            seq = 1
            if p == 0:
                continue  # warm-up pixel
            flat = np.asarray(flat, dtype=np.float64).reshape(int(n_wl), 4)
            flat_out[p - 1] = np.stack(
                [flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]], axis=0
            )
        f_handle = getattr(model_prof_tools, "f", None)
        if f_handle is not None and not f_handle.closed:
            f_handle.close()
        return flat_out.reshape(nx, ny, 4, int(n_wl))

    def run_inversion_cube(
        self,
        workdir: Path,
        logtau: np.ndarray,
        guess_T: np.ndarray,
        guess_v_los_kms: np.ndarray,
        guess_b_long_G: np.ndarray,
        observed_iquv: np.ndarray,
        n_wl: int,
        *,
        nodes_T: int = 8,
        nodes_V: int = 4,
        nodes_Bz: int = 4,
        wl_first: Optional[float] = None,
        wl_step_mA: Optional[float] = None,
        v_mic_cms: Optional[float] = None,
    ) -> dict:
        """Invert an ``(nx, ny)`` cube of observed profiles in one NICOLE run.

        Uses ``NICOLE.input_inversion.template`` with T, v_LOS and B_z nodes spread evenly
        over the guess grid and every other quantity fixed. The defaults 8/4/4 are the NICOLE
        manual's second cycle; for the full 1/1/1, 8/4/4, 10/6/4 progression, call this three
        times and pass each result as the next guess (per pixel). The same warm-up pixel as in
        :meth:`run_cube` is used. The atmosphere is read from NICOLE's nicole2.6 model records:
        T is block 2, v_LOS block 6 (cm/s) and B_long block 8, as checked against a known
        atmosphere run through :meth:`run_cube`.

        Args:
            workdir: Folder for NICOLE's files.
            logtau: Nodes, increasing.
            guess_T: Starting T in K, ``(nz,)`` for all pixels or ``(nx, ny, nz)``.
            guess_v_los_kms: Starting v_LOS in km/s, same shapes.
            guess_b_long_G: Starting B_long in G, same shapes.
            observed_iquv: ``(nx, ny, 4, n_wl)`` profiles to invert.
            n_wl: Number of wavelengths.
            nodes_T: Number of temperature nodes.
            nodes_V: Number of velocity nodes.
            nodes_Bz: Number of field nodes.
            wl_first: First wavelength in A (default: the configuration's).
            wl_step_mA: Step in mA (default: the configuration's).
            v_mic_cms: Microturbulence in cm/s (default: the configuration's).

        Returns:
            Dict with ``"T"`` (K), ``"v_los_kms"``, ``"b_long_G"`` (each ``(nx, ny, nz)``) and
            ``"fit_iquv"`` ``(nx, ny, 4, n_wl)``.

        Raises:
            ValueError: If ``observed_iquv`` or a guess has the wrong shape.
            RuntimeError: If NICOLE exits with an error.
            FileNotFoundError: If NICOLE writes no profile or model.

        Examples:
            >>> out = runner.run_inversion_cube(wd, logtau, T0, v0, b0, obs, 112)  # doctest: +SKIP
        """
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        observed_iquv = np.asarray(observed_iquv, dtype=np.float64)
        nx, ny = observed_iquv.shape[:2]
        if observed_iquv.shape[2] != 4 or observed_iquv.shape[3] != int(n_wl):
            raise ValueError(
                f"observed_iquv must be (nx, ny, 4, {n_wl}) (got {observed_iquv.shape})"
            )
        logtau = np.asarray(logtau, dtype=np.float64).reshape(-1)
        nz = logtau.size

        wl_first = self.cfg.wl_first if wl_first is None else wl_first
        wl_step_mA = self.cfg.wl_step_mA if wl_step_mA is None else wl_step_mA
        v_mic_cms = self.cfg.v_mic_cms if v_mic_cms is None else v_mic_cms

        npix = nx * ny

        def _flatten_guess(arr):
            # Accepts a single (nz,) atmosphere (broadcast to every pixel,
            # cycle 1) or a per-pixel (nx, ny, nz) atmosphere (chaining a
            # prior cycle's retrieved result in as this cycle's guess).
            """Guess as ``(npix, nz)``: tiled from ``(nz,)`` or reshaped from ``(nx, ny, nz)``."""
            arr = np.asarray(arr, dtype=np.float64)
            if arr.shape == (nz,):
                return np.tile(arr, (npix, 1))
            if arr.shape == (nx, ny, nz):
                return arr.reshape(npix, nz)
            raise ValueError(f"guess array must be ({nz},) or ({nx}, {ny}, {nz}) (got {arr.shape})")

        Tf = _flatten_guess(guess_T)
        Vf = _flatten_guess(guess_v_los_kms)
        Bf = _flatten_guess(guess_b_long_G)

        # Guess model cube, prepended with a warm-up pixel (T scaled) distinct
        # from pixel 0 -- mirrors run_cube's warm-up convention exactly.
        Tw = np.concatenate([Tf[:1] * 1.05, Tf], axis=0).reshape(1, npix + 1, nz)
        Vw = np.concatenate([Vf[:1], Vf], axis=0).reshape(1, npix + 1, nz)
        Bw = np.concatenate([Bf[:1], Bf], axis=0).reshape(1, npix + 1, nz)

        model_path = workdir / "model.model"
        write_binary_model_cube(
            model_path,
            logtau=logtau, T=Tw, v_los_kms=Vw, b_long_G=Bw,
            v_mic_cms=v_mic_cms, v_mac_cms=self.cfg.v_mac_cms,
            stray_light=self.cfg.stray_light, el_p_seed=self.cfg.el_p_seed,
        )

        # Observed profile cube: same (npix+1) sizing, warm-up record reuses
        # real pixel 0's observed profile (paired with the warm-up model
        # above; its inversion result is discarded regardless of content).
        obs_flat = observed_iquv.reshape(npix, 4, int(n_wl))
        obs_w = np.concatenate([obs_flat[:1], obs_flat], axis=0).reshape(1, npix + 1, 4, int(n_wl))
        obs_path = workdir / "observed.prof"
        write_prof_cube(obs_path, obs_w)

        template = self.inversion_template_path.read_text()
        rendered = template.format(
            NICOLE_COMMAND=str(self.nicole_command),
            INPUT_MODEL="model.model",
            OBSERVED_PROFILES="observed.prof",
            OUTPUT_PROFILE="profile.pro",
            OUTPUT_MODEL="model_out.mod",
            WL_FIRST=f"{wl_first:.4f}",
            WL_STEP_MA=f"{wl_step_mA:g}",
            N_WL=int(n_wl),
            NODES_T=int(nodes_T),
            NODES_V=int(nodes_V),
            NODES_BZ=int(nodes_Bz),
        )
        (workdir / "NICOLE.input").write_text(rendered)
        shutil.copyfile(self.lines_path, workdir / "LINES")
        link = workdir / "run_nicole.py"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(self.run_nicole_py)

        env = os.environ.copy()
        result = subprocess.run(
            [
                self.cfg.python_for_nicole,
                "./run_nicole.py",
                "--nicolecommand=" + str(self.nicole_command),
                "--modelin=model.model",
                "--profout=profile.pro",
            ],
            cwd=str(workdir),
            env=env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"NICOLE inversion failed in {workdir} (exit {result.returncode}).\n"
                f"--- STDOUT ---\n{result.stdout}\n--- STDERR ---\n{result.stderr}"
            )
        prof_path = workdir / "profile.pro"
        mod_path = workdir / "model_out.mod"
        for p in (prof_path, mod_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"NICOLE did not produce {p}.\n--- STDOUT ---\n{result.stdout}"
                )

        prof_ny = npix + 1
        fit_iquv = np.empty((npix, 4, int(n_wl)), dtype=np.float64)
        seq = 0
        for p in range(prof_ny):
            flat = read_prof(
                str(prof_path), "nicole2.3", 1, prof_ny, int(n_wl), 0, p, sequential=seq
            )
            seq = 1
            if p == 0:
                continue
            flat = np.asarray(flat, dtype=np.float64).reshape(int(n_wl), 4)
            fit_iquv[p - 1] = np.stack(
                [flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]], axis=0
            )
        f_handle = getattr(model_prof_tools, "f", None)
        if f_handle is not None and not f_handle.closed:
            f_handle.close()

        T_out = np.empty((npix, nz), dtype=np.float64)
        V_out = np.empty((npix, nz), dtype=np.float64)
        B_out = np.empty((npix, nz), dtype=np.float64)
        seq = 0
        for p in range(prof_ny):
            data = read_model(str(mod_path), "nicole2.6", 1, prof_ny, nz, 0, p, sequential=seq)
            seq = 1
            if p == 0:
                continue
            data = np.asarray(data, dtype=np.float64)
            T_out[p - 1] = data[2 * nz:3 * nz]
            V_out[p - 1] = data[6 * nz:7 * nz] / 1.0e5  # cm/s -> km/s
            B_out[p - 1] = data[8 * nz:9 * nz]
        f_handle = getattr(model_prof_tools, "f", None)
        if f_handle is not None and not f_handle.closed:
            f_handle.close()

        return {
            "T": T_out.reshape(nx, ny, nz),
            "v_los_kms": V_out.reshape(nx, ny, nz),
            "b_long_G": B_out.reshape(nx, ny, nz),
            "fit_iquv": fit_iquv.reshape(nx, ny, 4, int(n_wl)),
        }

    def run_pixels_from_h5(
        self,
        predictions_h5: Path,
        pixels: Optional[Sequence[tuple[int, int]]] = None,
        cleanup_workdirs: bool = True,
    ) -> Path:
        """Synthesize the pixels of a ``predictions.h5`` one by one and save ``syntheses.h5``.

        ``syntheses.h5`` holds ``pixels``, ``stokes_synth`` ``(n, 4, n_wl)`` (I, Q, U, V) and
        ``wavelengths``. Working folders are deleted only after the file is written.

        Args:
            predictions_h5: File from :meth:`PredictionExporter.export`.
            pixels: Subset of the exported pixels; ``None`` synthesizes all.
            cleanup_workdirs: Delete each pixel's working folder at the end.

        Returns:
            Path of ``syntheses.h5``.

        Raises:
            KeyError: If a requested pixel was not exported.
            RuntimeError: If NICOLE returns the wrong number of wavelengths.

        Examples:
            >>> NicoleRunner(cfg).run_pixels_from_h5(cfg.predictions_h5())  # doctest: +SKIP
        """
        with h5py.File(predictions_h5, "r") as h5:
            pred_pixels = h5["pixels"][...]
            logtau = h5["logtau"][...]
            T_all = h5["T"][...]
            Vz_all = h5["Vz"][...]
            Bz_all = h5["Bz"][...]
            wavelengths = h5["wavelengths"][...]
            Pgas_gt_all = h5["Pgas_gt"][...] if "Pgas_gt" in h5 else None

        if pixels is None:
            target_pixels = [tuple(p) for p in pred_pixels.tolist()]
        else:
            target_pixels = [tuple(p) for p in pixels]

        # Index map from pixel tuple -> row in predictions
        index_of = {tuple(p): i for i, p in enumerate(pred_pixels.tolist())}
        for p in target_pixels:
            if p not in index_of:
                raise KeyError(
                    f"Pixel {p} was not exported in {predictions_h5}. "
                    f"Available: {list(index_of)[:10]}{'...' if len(index_of) > 10 else ''}"
                )

        n = len(target_pixels)
        synth = np.full((n, 4, self.cfg.n_wl), np.nan, dtype=np.float64)
        for k, (ix, iy) in enumerate(target_pixels):
            row = index_of[(ix, iy)]
            workdir = self.prepare_workdir(
                ix=int(ix),
                iy=int(iy),
                logtau=logtau,
                T=T_all[row],
                v_los_kms=Vz_all[row],
                b_long_G=Bz_all[row],
                gt_pressure=Pgas_gt_all[row] if Pgas_gt_all is not None else None,
            )
            iquv = self.run_pixel(workdir)
            if iquv.shape[1] != self.cfg.n_wl:
                raise RuntimeError(
                    f"NICOLE returned {iquv.shape[1]} wavelengths for pixel "
                    f"({ix},{iy}), expected {self.cfg.n_wl}"
                )
            synth[k] = iquv

        out_path = self.cfg.syntheses_h5()
        if out_path.exists():
            out_path.unlink()
        with h5py.File(out_path, "w") as h5:
            h5.create_dataset(
                "pixels",
                data=np.asarray(target_pixels, dtype=np.int64),
            )
            h5.create_dataset("stokes_synth", data=synth)
            h5.create_dataset("wavelengths", data=wavelengths)
            h5.attrs["source"] = self.cfg.source
            h5.attrs["experiment_root"] = self.cfg.experiment_root
            h5.attrs["model_type"] = self.cfg.model_type
            h5.attrs["region_label"] = self.cfg.region_label
            h5.attrs["output_root"] = str(self.cfg.output_root)
            if self.cfg.source == "muram":
                h5.attrs["muram_step"] = int(self.cfg.muram_step)
                h5.attrs["add_gt_pressure"] = bool(self.cfg.add_gt_pressure)

        if cleanup_workdirs:
            for ix, iy in target_pixels:
                workdir = self._pixel_workdir(int(ix), int(iy))
                _rmtree_retry(workdir)

        return out_path


def _format_bin_range(lo: float, hi: float) -> str:
    """Folder-name label of a field bin, 4 significant figures.

    Examples:
        >>> _format_bin_range(12.345, 1234.5)
        '12.35-1234'
    """
    return f"{lo:.4g}-{hi:.4g}"


class SynthesisComparator:
    """Compare synthesized and input profiles of the same pixels: chi-square and overlay plots."""

    def __init__(self, cfg: SynthesisConfig):
        """Set up an empty comparator.

        Args:
            cfg: Synthesis configuration (locates the pixel-selection bins).
        """
        self.cfg = cfg
        self.predictions_h5: Optional[Path] = None
        self.syntheses_h5: Optional[Path] = None
        self._pred_pixels = None
        self._synth_pixels = None
        self._stokes_obs = None
        self._stokes_synth = None
        self._wavelengths = None
        self._bz = None
        self._bin_info: dict[tuple[int, int], dict] = {}

    def load(self, predictions_h5: Path, syntheses_h5: Path) -> None:
        """Read the input profiles, predicted Bz and synthesized profiles, and the pixels' field bins.

        Args:
            predictions_h5: File from :meth:`PredictionExporter.export`.
            syntheses_h5: File from :meth:`NicoleRunner.run_pixels_from_h5`.

        Examples:
            >>> comp = SynthesisComparator(cfg); comp.load(cfg.predictions_h5(), cfg.syntheses_h5())  # doctest: +SKIP
        """
        self.predictions_h5 = predictions_h5
        self.syntheses_h5 = syntheses_h5
        with h5py.File(predictions_h5, "r") as h5:
            self._pred_pixels = h5["pixels"][...]
            self._stokes_obs = h5["stokes_obs"][...]
            self._wavelengths = h5["wavelengths"][...]
            self._bz = h5["Bz"][...]
        with h5py.File(syntheses_h5, "r") as h5:
            self._synth_pixels = h5["pixels"][...]
            self._stokes_synth = h5["stokes_synth"][...]

        # Pick up the |B_LOS| stratification bins from sample_pixels.py, if that
        # step was run for this experiment/model/region (utils/pixel_sampling.py).
        self._bin_info = {}
        selection_json = self.cfg.out_dir() / "pixel_selection" / "selected_pixels.json"
        if selection_json.exists():
            with open(selection_json) as f:
                selection = json.load(f)
            bin_edges = selection["bin_edges_gauss"]
            for entry in selection["pixels"]:
                b = entry["bin"]
                self._bin_info[(entry["ix"], entry["iy"])] = {
                    "bin_lo": bin_edges[b],
                    "bin_hi": bin_edges[b + 1],
                    "abs_bz_gauss": entry["abs_bz_gauss"],
                }

    def _abs_b_los(self, ix: int, iy: int) -> float:
        """|B_LOS| of a pixel in G: from the pixel selection, else the prediction at the deepest node."""
        info = self._bin_info.get((ix, iy))
        if info is not None:
            return info["abs_bz_gauss"]
        return float(np.abs(self._bz[self._pred_row(ix, iy), -1]))

    def _bin_range(self, ix: int, iy: int) -> Optional[tuple[float, float]]:
        """Field bin ``(lo, hi)`` of a pixel in G, or ``None`` if it was not stratified."""
        info = self._bin_info.get((ix, iy))
        if info is None:
            return None
        return info["bin_lo"], info["bin_hi"]

    def _pred_row(self, ix: int, iy: int) -> int:
        """Row of a pixel in ``predictions.h5``."""
        for i, (a, b) in enumerate(self._pred_pixels.tolist()):
            if a == ix and b == iy:
                return i
        raise KeyError(f"Pixel ({ix},{iy}) not in predictions.h5")

    def _synth_row(self, ix: int, iy: int) -> int:
        """Row of a pixel in ``syntheses.h5``."""
        for i, (a, b) in enumerate(self._synth_pixels.tolist()):
            if a == ix and b == iy:
                return i
        raise KeyError(f"Pixel ({ix},{iy}) not in syntheses.h5")

    def chi_square(
        self,
        sigma_I: float = 1e-3,
        sigma_V: float = 1e-3,
    ) -> dict:
        """Chi-square of I and V per pixel: ``sum(((obs - synth) / sigma)^2)`` over wavelength.

        The input file stores I and V (channels 0 and 1); the synthesis stores I, Q, U, V
        (V is channel 3). The sums are not divided by the number of points.

        Args:
            sigma_I: Noise of I in continuum units.
            sigma_V: Noise of V in continuum units.

        Returns:
            ``{"ix,iy": {"chi2_I", "chi2_V", "n_wl"}}``.

        Examples:
            >>> chi = comp.chi_square(sigma_I=1e-3, sigma_V=1e-3)  # doctest: +SKIP
        """
        out = {}
        for ix, iy in self._synth_pixels.tolist():
            i_obs = self._stokes_obs[self._pred_row(ix, iy), 0]
            v_obs = self._stokes_obs[self._pred_row(ix, iy), 1]
            i_syn = self._stokes_synth[self._synth_row(ix, iy), 0]
            v_syn = self._stokes_synth[self._synth_row(ix, iy), 3]
            chi2_I = float(np.nansum(((i_obs - i_syn) / sigma_I) ** 2))
            chi2_V = float(np.nansum(((v_obs - v_syn) / sigma_V) ** 2))
            out[f"{ix},{iy}"] = {
                "chi2_I": chi2_I,
                "chi2_V": chi2_V,
                "n_wl": int(np.isfinite(i_syn).sum()),
            }
        return out

    def plot_overlay(self, ix: int, iy: int, out_dir: Path) -> Path:
        """Plot input and synthesized I and V of one pixel, saved under its field-bin folder.

        Args:
            ix: Pixel index along the first axis.
            iy: Pixel index along the second axis.
            out_dir: Base folder; the figure goes into a ``<lo>-<hi>`` subfolder when the
                pixel has a field bin.

        Returns:
            Path of the PNG.

        Examples:
            >>> comp.plot_overlay(100, 200, Path("figs"))  # doctest: +SKIP
        """
        import matplotlib.pyplot as plt

        bin_range = self._bin_range(ix, iy)
        target_dir = out_dir / _format_bin_range(*bin_range) if bin_range is not None else out_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        i_obs = self._stokes_obs[self._pred_row(ix, iy), 0]
        v_obs = self._stokes_obs[self._pred_row(ix, iy), 1]
        synth_row = self._stokes_synth[self._synth_row(ix, iy)]
        i_syn, q_syn, u_syn, v_syn = synth_row[0], synth_row[1], synth_row[2], synth_row[3]
        wl = self._wavelengths
        abs_b_los = self._abs_b_los(ix, iy)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(wl, i_obs, color="k", label="Observed")
        axes[0].plot(wl, i_syn, color="C3", ls="--", label="NICOLE (MUISCA atm.)")
        axes[0].set_title(f"Stokes I  pix=({ix},{iy})")
        axes[0].set_xlabel("Wavelength [Å]")
        axes[0].set_ylabel("I / I_c")
        axes[0].legend()

        axes[1].plot(wl, v_obs, color="k", label="Observed")
        axes[1].plot(wl, v_syn, color="C3", ls="--", label="NICOLE")
        axes[1].set_title(f"Stokes V  pix=({ix},{iy})")
        axes[1].set_xlabel("Wavelength [Å]")
        axes[1].set_ylabel("V / I_c")
        axes[1].legend()

        fig.suptitle(f"|B_LOS| = {abs_b_los:.3g} G")
        plt.tight_layout()
        out_path = target_dir / f"overlay_pix_{ix:05d}_{iy:05d}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out_path
