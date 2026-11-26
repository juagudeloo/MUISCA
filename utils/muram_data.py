"""MURaM simulation data: atmospheres, synthetic Stokes profiles and the per-step training Dataset.

Two data sources use this module.

* ``nicole_tau500`` (current). ``scripts/synthesis/generate_tau500_stokes.py`` reads the raw
  cube with :meth:`MhdData.load_step`, builds its own tau_500 scale
  (``utils/tau500_opacity.py``) and has NICOLE synthesize the profiles. Training then uses
  :class:`StokesData` only as a container for the Hinode degradation (LSF, resampling) and
  :class:`MuramStepDataset` to serve normalized pixels.
* ``muram_legacy``. :meth:`MhdData.compute_optical_depth` integrates a Rosseland-mean opacity
  table, :meth:`MhdData.remap_to_optical_depth` moves T, Vz and Bz onto that scale and
  :meth:`StokesData.load_stokes` reads the older ``stokes_<step>.npy`` cubes.

The module also holds the pixel-selection helpers used during training: granulation and
polarization region masks, region balancing and |B_LOS|-strength balancing.

Sign conventions (verified on step 110): MURaM Vz is positive upward (granules have
Vz > 0) and Bz is positive upward, i.e. toward the observer at disk centre.

Examples:
    Typical training-side use with the ``nicole_tau500`` source (needs the cluster files)::

        stokes, mhd = load_source_arrays(step=110, config=TrainingConfig())
        stokes.load_hinode_lsf(config.data_path / config.lsf_path)
        stokes.apply_spectral_convolution()
        stokes.resample_to_hinode()
        dataset = MuramStepDataset(stokes.data, mhd, stokes_norm, mhd_norm)
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
from scipy.ndimage import convolve1d
from scipy.interpolate import RegularGridInterpolator, interp1d
from utils.hinode_wavelengths import hinode_wavelength_grid
from scipy.integrate import cumulative_trapezoid
import torch
from torch.utils.data import Dataset

from utils.normalizer import MhdNormalizer, StokesNormalizer


class MhdData:
    """Load a MURaM snapshot and, on the legacy path, put it on an optical-depth scale.

    Arrays are ``(nx, ny, nz)`` astropy Quantities in ``self.data`` (geometric height,
    z index increasing upward, 10 km per layer) and ``(nx, ny, n_logtau)`` Quantities in
    ``self.od_data`` after :meth:`remap_to_optical_depth`.

    Attributes:
        data: ``T`` (K), ``P`` (dyn cm^-2), ``rho`` (g cm^-3), ``Vz`` (km/s) and ``Bz`` (G).
        logtau: log10 tau per cell, ``(nx, ny, nz)``, after :meth:`compute_optical_depth`.
        od_data: Quantities remapped to the requested log tau grid.
        z0: Index of the layer whose horizontal mean T is closest to 5780 K.
        height_levels: Height of each layer relative to ``z0``, in km.
    """
    
    def __init__(self, data_path: str, nx: int = 480, ny: int = 480, nz: int = 256):
        """Set up an empty container for one MURaM run.

        Args:
            data_path: Folder holding the ``eos.NNN000`` and ``result_k.NNN000`` files.
            nx: Number of cells along x in the files (480 for this run, 25 km each).
            ny: Number of cells along y.
            nz: Number of layers along the vertical in the files (256, 10 km each).
        """
        self.data_path = Path(data_path)
        self.nx = nx
        self.ny = ny
        self.nz = nz  # Original grid z dimension
        self.z_max: Optional[int] = None  # Trimmed z dimension (if z_max was applied during load_step)
        
        # Data containers
        self.data: Dict[str, np.ndarray] = {}
        self.logtau: Optional[np.ndarray] = None
        self.od_data: Dict[str, np.ndarray] = {}
        
        # Reference layers
        self.z0: Optional[int] = None  # Reference height for T ~ 5780 K
        self.height_levels: Optional[np.ndarray] = None
        
        # Opacity interpolator
        self.kappa_interp: Optional[RegularGridInterpolator] = None
    
    @staticmethod
    def to_value(quantity, target_unit):
        """Return the number of ``quantity`` in ``target_unit``, or ``quantity`` itself if it has no unit.

        Args:
            quantity: An astropy Quantity, or a plain number already in ``target_unit``.
            target_unit: Unit to convert to, e.g. ``u.cm``.

        Returns:
            The magnitude in ``target_unit`` (plain numbers pass through unchanged).

        Examples:
            >>> import astropy.units as u
            >>> MhdData.to_value(10 * u.km, u.cm)
            1000000.0
            >>> MhdData.to_value(2.5, u.cm)
            2.5
        """
        if hasattr(quantity, 'to'):
            return quantity.to(target_unit).value
        return quantity
        
    def load_step(self, step: Union[int, str], z_max: Optional[int] = None,
                  read_horizontal_field: bool = False) -> Dict[str, np.ndarray]:
        """Read one snapshot (T, P, rho, Vz, Bz) from the raw MURaM binaries.

        Reads ``eos`` (T, P), ``result_0`` (rho), ``result_2`` (rho times the vertical
        velocity) and ``result_6`` (vertical field). The files are float32 in
        ``(nx, nz, ny)`` order and are transposed to ``(nx, ny, nz)``. Derived fields are
        ``Vz = result_2 / rho / 1e5`` (cm/s to km/s) and ``Bz = result_6 * sqrt(4 pi)``
        (code units to G). Also sets ``z0`` and ``height_levels``.

        Note:
            Vz and Bz are positive upward (toward the observer at disk centre). NICOLE's
            v_LOS is positive for a redshift, so the synthesis hands it ``-Vz``. The
            horizontal field is read from ``result_7`` (Bx) and ``result_5`` (By), verified
            with div B, when ``read_horizontal_field`` is set.

        Args:
            step: Snapshot number. ``110``, ``"110"`` and ``"110000"`` all select the files
                ending in ``.110000``.
            z_max: Keep only layers ``0 .. z_max - 1`` (drops the top of the box). ``None``
                keeps all ``nz`` layers.
            read_horizontal_field: Also read Bx (``result_7``) and By (``result_5``), needed
                to synthesize Q and U.

        Returns:
            ``self.data``, a dict of ``(nx, ny, nz)`` Quantities: ``T`` (K), ``P``
            (dyn cm^-2), ``rho`` (g cm^-3), ``Vz`` (km/s) and ``Bz`` (G), plus ``Bx`` and
            ``By`` (G) with ``read_horizontal_field``.

        Raises:
            FileNotFoundError: If one of the four required files is missing, or, with
                ``read_horizontal_field``, one of ``result_5`` and ``result_7``.
            ValueError: If ``z_max`` is not in ``1 .. nz``.

        Examples:
            >>> mhd = MhdData("data/muram-simulation/")  # doctest: +SKIP
            >>> data = mhd.load_step(110)                 # doctest: +SKIP
            >>> data["T"].shape, data["Bz"].unit          # doctest: +SKIP
            ((480, 480, 256), Unit("G"))
        """
        # Normalize step -> suffix "NNN000"
        if isinstance(step, int):
            suffix = f"{step:03d}000"
        else:
            s = str(step)
            if s.isdigit() and len(s) == 3:
                suffix = f"{int(s):03d}000"
            elif s.isdigit() and len(s) == 6:
                suffix = s
            else:
                suffix = s
        
        def _path(name):
            """Path of file ``name`` for this step, e.g. ``result_0.110000``."""
            return self.data_path / f"{name}.{suffix}"
        
        # Check required files
        required = ["eos", "result_0", "result_2", "result_6"]
        missing = [r for r in required if not _path(r).exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for step {suffix}: {missing}")
        
        print(f"Loading step {suffix}")
        
        # Read EOS (2, nx, nz, ny)
        mtpr = np.fromfile(str(_path("eos")), dtype=np.float32)
        mtpr = mtpr.reshape((2, self.nx, self.nz, self.ny), order="C")
        mtpr = mtpr.transpose(0, 1, 3, 2)
        T = mtpr[0, :, :, :]
        P = mtpr[1, :, :, :]
        
        # Helper to read 3D fields (nx, nz, ny)
        def _read3(name):
            """Read a float32 ``(nx, nz, ny)`` file and return it as ``(nx, ny, nz)``."""
            arr = np.fromfile(str(_path(name)), dtype=np.float32)
            arr = arr.reshape((self.nx, self.nz, self.ny), order="C")
            return arr.transpose(0, 2, 1)
        
        rho = _read3("result_0")
        rho_Vz = _read3("result_2")
        Bz = _read3("result_6")
        # Horizontal field: result_7 is along x and result_5 along y (checked with div B,
        # 0.002 against 0.055 for the swapped pair). Only read when asked for (the Stokes
        # synthesis needs it for Q and U; other callers would just pay two extra cubes).
        Bx = By = None
        if read_horizontal_field:
            have = [_path(n).exists() for n in ("result_7", "result_5")]
            if not all(have):
                raise FileNotFoundError(
                    f"Step {suffix}: the horizontal field needs both result_7 (Bx) and "
                    f"result_5 (By); found result_7={have[0]}, result_5={have[1]}"
                )
            Bx = _read3("result_7")
            By = _read3("result_5")
        
        # Derived fields
        Vz = rho_Vz / rho / 1e5  # convert to km/s
        coef = np.sqrt(4.0 * 3.14159265)
        Bz = Bz * coef
        Bx = None if Bx is None else Bx * coef
        By = None if By is None else By * coef
        
        # Optionally trim top z layers
        # Keep self.nz unchanged for file I/O. Track trimmed size in self.z_max.
        if z_max is not None:
            if not (0 < z_max <= self.nz):
                raise ValueError(f"z_max must be in 1..{self.nz}, got {z_max}")
            T = T[:, :, :z_max]
            P = P[:, :, :z_max]
            rho = rho[:, :, :z_max]
            Vz = Vz[:, :, :z_max]
            Bz = Bz[:, :, :z_max]
            Bx = None if Bx is None else Bx[:, :, :z_max]
            By = None if By is None else By[:, :, :z_max]
            self.z_max = z_max
            print(f"  Trimmed z axis to first {z_max} layers (0..{z_max-1})")
        else:
            self.z_max = self.nz
        
        self.data = {
            "T": T*u.K,
            "P": P*u.dyn/u.cm**2,
            "rho": rho*u.g/u.cm**3,
            "Vz": Vz*u.km/u.s,
            "Bz": Bz*u.G,
        }
        if read_horizontal_field:
            self.data["Bx"] = Bx * u.G
            self.data["By"] = By * u.G
        
        # Compute reference layer (z0) and height levels
        self._compute_reference_layer()
        
        print("  Done.")
        return self.data
    
    def _compute_reference_layer(self, height_step_km: float = 10.0):
        """Set ``z0`` (layer with mean T closest to 5780 K) and ``height_levels`` in km relative to it."""
        if "T" not in self.data:
            raise ValueError("Temperature data not loaded")
        
        mean_T_z = np.nanmean(self.data["T"], axis=(0, 1))
        self.z0 = int(np.argmin(np.abs(mean_T_z - 5780.0*u.K)))
        # Use actual data z-size (self.z_max) for height levels, not original nz
        nz_actual = self.data["T"].shape[2] if "T" in self.data else self.z_max
        self.height_levels = np.arange(0, nz_actual, 1) * height_step_km - self.z0 * height_step_km
        print(f"  Reference layer (T ~ 5780 K) at z-index {self.z0}")
    
    def load_opacity_table(self, kappa_path: str):
        """Build the Rosseland-mean opacity interpolator kappa(log P, log T) (legacy path only).

        The table ``kappa.0.dat`` has three columns: temperature index, pressure index and
        opacity in cm^2 g^-1. The indices refer to a fixed grid of 56 values of log10 T (K)
        from 3.32 to 5.30 and 21 values of log10 P (dyn cm^-2) from -2 to 8. The
        interpolation is linear in (log P, log T).

        Args:
            kappa_path: Path to ``kappa.0.dat`` (in the repository at ``data/csv/``).

        Examples:
            >>> import contextlib, io
            >>> mhd = MhdData(".")
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     mhd.load_opacity_table("data/csv/kappa.0.dat")  # doctest: +SKIP
            >>> float(mhd.kappa_interp([[5.0, 3.76]])[0])  # doctest: +SKIP
            0.75...
        """
        # Temperature and pressure grid from opacity table
        tab_T = np.array([3.32, 3.34, 3.36, 3.38, 3.40, 3.42, 3.44, 3.46, 3.48, 3.50,
                          3.52, 3.54, 3.56, 3.58, 3.60, 3.62, 3.64, 3.66, 3.68, 3.70,
                          3.73, 3.76, 3.79, 3.82, 3.85, 3.88, 3.91, 3.94, 3.97, 4.00,
                          4.05, 4.10, 4.15, 4.20, 4.25, 4.30, 4.35, 4.40, 4.45, 4.50,
                          4.55, 4.60, 4.65, 4.70, 4.75, 4.80, 4.85, 4.90, 4.95, 5.00,
                          5.05, 5.10, 5.15, 5.20, 5.25, 5.30])  # log(T) in K
        
        tab_p = np.array([-2., -1.5, -1., -0.5, 0., 0.5, 1., 1.5, 2., 2.5,
                          3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8.])  # log(P) in dyne/cm^2
        
        df_kappa = pd.read_csv(kappa_path, sep=r'\s+', header=None)
        df_kappa.columns = ["Temperature index", "Pressure index", "Opacity value"]
        
        temp_indices = df_kappa["Temperature index"].unique()
        press_indices = df_kappa["Pressure index"].unique()
        opacity_values = df_kappa.pivot(
            index="Pressure index",
            columns="Temperature index",
            values="Opacity value"
        ).values
        
        Tk = tab_T[temp_indices]
        Pk = tab_p[press_indices]
        K = opacity_values * u.cm**2 / u.g  # Rosseland mean opacity in cm²/g
        
        self.kappa_interp = RegularGridInterpolator((Pk, Tk), K, method="linear")
        print(f"Opacity interpolator loaded from {kappa_path}")
        print(f"  T range: [{Tk.min():.3f}, {Tk.max():.3f}] (log10 K)")
        print(f"  P range: [{Pk.min():.3f}, {Pk.max():.3f}] (log10 dyne/cm²)")
    
    def compute_optical_depth(self, dz=10*u.km, baseline: float = 1e-5) -> np.ndarray:
        """Integrate tau(z) = integral of kappa rho dz from the top of the box down (legacy path only).

        kappa comes from :meth:`load_opacity_table` with T and P clipped to the table range.
        The integral uses the trapezoidal rule along z from the top layer (tau = 0) to the
        bottom, and tau is clipped at 1e-30 before taking log10, so the top layer has
        log tau = -30.

        Note:
            This Rosseland-mean scale is not the tau_500 scale used by the ``nicole_tau500``
            data, which ``generate_tau500_stokes.py`` builds from the H- opacity at 5000 A.

        Args:
            dz: Layer thickness, as a Quantity (default 10 km) or a number in cm.
            baseline: Unused; kept for backward compatibility.

        Returns:
            log10 tau, ``(nx, ny, nz)``, also stored in ``self.logtau``.

        Raises:
            ValueError: If the opacity table or T, P, rho are not loaded, or if a
                non-positive tau appears.

        Examples:
            >>> mhd.load_opacity_table("data/csv/kappa.0.dat")  # doctest: +SKIP
            >>> logtau = mhd.compute_optical_depth(dz=10 * u.km)  # doctest: +SKIP
        """
        if self.kappa_interp is None:
            raise ValueError("Opacity interpolator not loaded. Call load_opacity_table() first.")
        
        if "T" not in self.data or "P" not in self.data or "rho" not in self.data:
            raise ValueError("T, P, and rho must be loaded")
        
        print("Computing optical depth...")
        
        # Convert dz to cm if it's a Quantity
        dz_cm = self.to_value(dz, u.cm)
        
        # Prepare T and P in log10, clipped to interpolation bounds
        T_log = np.log10(self.data["T"].value)
        P_log = np.log10(self.data["P"].value)
        
        # Get interpolator bounds
        Pk_min, Pk_max = self.kappa_interp.grid[0].min(), self.kappa_interp.grid[0].max()
        Tk_min, Tk_max = self.kappa_interp.grid[1].min(), self.kappa_interp.grid[1].max()
        
        T_log = np.clip(T_log, Tk_min + 1e-5, Tk_max - 1e-5)
        P_log = np.clip(P_log, Pk_min + 1e-5, Pk_max - 1e-5)
        
        # Stack for interpolation
        PT_log = np.stack((P_log.flatten(), T_log.flatten()), axis=-1)
        
        # Interpolate opacity and multiply by density with units
        kappa_rho_interp = self.kappa_interp(PT_log)
        kappa_rho_interp = kappa_rho_interp.reshape(self.data["T"].shape)*(u.cm**2/u.g)
        # κ*ρ: (cm²/g) * (g/cm³) = cm⁻¹ (dimensionless for optical depth integration)
        # Extract numeric value for numerical integration
        kappa_rho = (kappa_rho_interp * self.data["rho"]).to(u.cm**-1).value
        
        # Integrate from top to bottom (reverse z, integrate, reverse back)
        kappa_rev = kappa_rho[:, :, ::-1]
        tau_rev = cumulative_trapezoid(kappa_rev, dx=dz_cm, axis=2, initial=0.0)
        tau = tau_rev[:, :, ::-1]
        tau = np.clip(tau, a_min=1e-30, a_max=None)
        
        # Check for non-positive values
        nonpos_mask = tau <= 0
        if np.any(nonpos_mask):
            cnt = int(np.count_nonzero(nonpos_mask))
            raise ValueError(f"Non-positive tau encountered: {cnt} elements <= 0")
        
        self.logtau = np.log10(tau)
        print("  Optical depth computed.")
        return self.logtau
    
    def remap_to_optical_depth(self, new_logtau: np.ndarray, 
                                quantities: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Move quantities from geometric height to a log tau grid, column by column.

        Each (x, y) column is linearly interpolated in log tau (``interp1d``, linear,
        extrapolating outside the column's range).

        Args:
            new_logtau: Target log tau values, e.g. ``np.arange(-3.0, 1.45, 0.1)``.
            quantities: Keys of ``self.data`` to remap. Default ``["T", "Vz", "Bz"]``.
                Missing keys are skipped with a message.

        Returns:
            ``self.od_data``: Quantities of shape ``(nx, ny, len(new_logtau))`` with the same
            units as in ``self.data``.

        Raises:
            ValueError: If :meth:`compute_optical_depth` has not been run.

        Examples:
            >>> import contextlib, io
            >>> mhd = MhdData(".", nx=1, ny=1, nz=4)
            >>> mhd.logtau = np.array([[[1.0, 0.0, -1.0, -2.0]]])          # z up, tau down
            >>> mhd.data = {"T": np.array([[[7000.0, 6000.0, 5000.0, 4000.0]]]) * u.K}
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     od = mhd.remap_to_optical_depth(np.array([-1.5, 0.5]), ["T"])
            >>> od["T"].value.ravel().tolist(), od["T"].unit
            ([4500.0, 6500.0], Unit("K"))
        """
        if self.logtau is None:
            raise ValueError("Optical depth not computed. Call compute_optical_depth() first.")
        
        if quantities is None:
            quantities = ["T", "Vz", "Bz"]
        
        n_logtau = new_logtau.shape[0]
        self.od_data = {}
        
        print(f"Remapping to optical depth coordinates ({n_logtau} levels)...")
        
        for q in quantities:
            if q not in self.data:
                print(f"  Skipping {q}: not in loaded data")
                continue
            
            print(f"  Processing {q}")
            geom_atm = self.data[q]
            new_quantity = np.zeros((self.nx, self.ny, n_logtau))
            
            for ix in range(self.nx):
                for iy in range(self.ny):
                    mapper = interp1d(
                        x=self.logtau[ix, iy, :],
                        y=geom_atm.value[ix, iy, :],
                        kind="linear",
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    new_quantity[ix, iy, :] = mapper(new_logtau)
            
            self.od_data[q] = new_quantity*u.Unit(geom_atm.unit)
        
        print("  Remapping complete.")
        return self.od_data
    
    def plot_surfaces(self, z_idx: int, data_source: str = "geometric",
                      zero_height: bool = True, pixel_size_km: float = 25,
                      quantities: Optional[List[str]] = None,
                      second_plot: str = "profile",
                      default_cmap: str = "viridis",
                      limit_values: Optional[Dict[str, Tuple[float, float]]] = None,
                      save_dir: Optional[str] = None, dpi: int = 150,
                      return_figs: bool = False):
        """Plot a horizontal map of each quantity at one layer, next to its mean profile or histogram.

        rho and P are shown in log10. Signed quantities (B, v) get a symmetric colour range
        from their 1st and 99th percentiles unless ``limit_values`` is given.

        Args:
            z_idx: Layer (geometric) or log tau index (optical depth) to map.
            data_source: ``"geometric"`` for ``self.data`` or ``"optical_depth"`` for
                ``self.od_data``.
            zero_height: Mark the T = 5780 K reference height on the profile panel.
            pixel_size_km: Horizontal cell size for the scale bar, in km.
            quantities: Keys to plot. Default: all keys of the chosen data.
            second_plot: ``"profile"`` (horizontal mean against height) or ``"histogram"``
                (values at ``z_idx``).
            default_cmap: Colormap for quantities without a preset one.
            limit_values: ``{key: (vmin, vmax)}`` colour limits in the quantity's units.
            save_dir: Save PNGs there instead of showing them.
            dpi: Resolution of saved figures.
            return_figs: Return the figures instead of showing or saving them.

        Returns:
            A list of ``(key, Figure)`` if ``return_figs``; otherwise ``None``.

        Raises:
            ValueError: For an unknown ``data_source`` or ``second_plot``.

        Examples:
            >>> figs = mhd.plot_surfaces(z_idx=190, quantities=["T"], return_figs=True)  # doctest: +SKIP
        """
        if data_source == "geometric":
            data = self.data
            height_levels = self.height_levels
        elif data_source == "optical_depth":
            data = self.od_data
            height_levels = None  # Will be set from caller
        else:
            raise ValueError("data_source must be 'geometric' or 'optical_depth'")
        
        if quantities is None:
            quantities = list(data.keys())
        
        cmap_map = {
            "T": "hot", "P": "summer", "rho": "spring",
            "Bx": "PRGn", "By": "PRGn", "Bz": "PiYG",
            "Vx": "managua", "Vy": "managua", "Vz": "bwr_r",
        }
        
        latex_map = {
            "T": r"$T$", "P": r"$P$", "rho": r"$\rho$",
            "Vx": r"$v_x$", "Vy": r"$v_y$", "Vz": r"$v_z$",
            "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$",
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in data:
                print(f"  Skipping {q}: not in data")
                continue
            
            arr = data[q]
            q_unit = arr.unit.to_string("latex") if hasattr(arr, 'unit') else ""
            arr = arr.value if hasattr(arr, 'value') else arr
            if arr.ndim != 3:
                print(f"  Skipping {q}: not 3D")
                continue
            
            surf = arr[:, :, z_idx]
            profile = np.nanmean(arr, axis=(0, 1))
            
            cmap_name = cmap_map.get(q, default_cmap)
            cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap(default_cmap)
            
            # Determine color limits
            if limit_values and q in limit_values:
                vmin, vmax = limit_values[q]
            elif q in ["Bx", "By", "Bz", "Vz", "Vx", "Vy"]:
                vmin, vmax = np.quantile(surf, [0.01, 0.99])
                maxabs = max(abs(vmin), abs(vmax))
                vmin, vmax = -maxabs, maxabs
            else:
                vmin, vmax = None, None
            
            q_label = latex_map.get(q, q)
            
            # Log transform for rho and P
            if q in ["rho", "P"]:
                profile = np.log10(profile)
                surf = np.log10(surf)
                q_label = q_label + " (log10)"
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                
                im = axs[0].imshow(surf.T, origin="lower", cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
                axs[0].set_title(rf"{q_label} surface (z={z_idx})", fontsize=12)
                axs[0].set_axis_off()
                
                scalebar = ScaleBar(pixel_size_km, 'km', length_fraction=0.25, location='lower left', pad=0.5)
                axs[0].add_artist(scalebar)
                plt.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04, label=f"{q_unit}")
                
                if second_plot == "profile":
                    line_color = cmap(0.6)
                    
                    if height_levels is not None:
                        axs[1].plot(height_levels, profile, "-o", ms=3, color=line_color)
                        if self.z0 is not None and zero_height and data_source == "geometric":
                            axs[1].axvline(z_idx * 10 - self.z0 * 10, color="gray", lw=1, ls="--")
                        axs[1].set_xlabel("Height (km)")
                    else:
                        x_vals = np.arange(len(profile))
                        axs[1].plot(x_vals, profile, "-o", ms=3, color=line_color)
                        axs[1].set_xlabel("Z-index")
                    
                    if zero_height and data_source == "geometric":
                        y_min, y_max = axs[1].get_ylim()
                        y_mid = 0.5 * (y_min + y_max)
                        axs[1].text(0, y_mid, r"$T \approx 5780$ K",
                                    color='red', fontsize=9, ha='center', va='center',
                                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
                        axs[1].axvline(0, color='red', lw=1.5, ls='-')
                    
                    axs[1].set_ylabel(f"mean {q_label} (x,y)")
                    axs[1].set_title(rf"{q_label} profile along z", fontsize=12)
                
                elif second_plot == "histogram":
                    axs[1].hist(surf.ravel(), bins=50, color="black", alpha=0.7)
                    axs[1].set_xlabel(f"{q_label}")
                    axs[1].set_ylabel("Counts")
                    axs[1].set_title(rf"{q_label} histogram at z={z_idx}", fontsize=12)
                
                else:
                    raise ValueError(f"Unknown second_plot option: {second_plot}. Must be 'profile' or 'histogram'")
                
                plt.tight_layout()
                
                if return_figs:
                    figs.append((q, fig))
                else:
                    if save_path is not None:
                        fname = f"{q}_surface_z{z_idx}.png"
                        fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                        print(f"Saved: {save_path / fname}")
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)
        
        if return_figs:
            return figs
    
    def plot_slices(self, x_idx: int, pixel_size_km: float = 25,
                    quantities: Optional[List[str]] = None,
                    default_cmap: str = "viridis", dz_km: float = 10,
                    figsize: Tuple[int, int] = (7, 8),
                    save_dir: Optional[str] = None, dpi: int = 150,
                    return_figs: bool = False):
        """Plot vertical (y, z) slices at one x, with the log tau = 0 contour and the T = 5780 K layer.

        Args:
            x_idx: Column index along x.
            pixel_size_km: Horizontal cell size for the scale bar, in km.
            quantities: Keys of ``self.data`` to plot. Default: all.
            default_cmap: Colormap for quantities without a preset one.
            dz_km: Layer thickness in km, for the height ticks when ``height_levels`` is unset.
            figsize: Figure size in inches.
            save_dir: Save PNGs there instead of showing them.
            dpi: Resolution of saved figures.
            return_figs: Return the figures instead of showing or saving them.

        Returns:
            A list of ``(key, Figure)`` if ``return_figs``; otherwise ``None``.

        Examples:
            >>> figs = mhd.plot_slices(x_idx=120, quantities=["Vz"], return_figs=True)  # doctest: +SKIP
        """
        if quantities is None:
            quantities = list(self.data.keys())
        
        cmap_map = {
            "T": "hot", "P": "summer", "rho": "spring",
            "Bx": "PRGn", "By": "PRGn", "Bz": "PiYG",
            "Vx": "managua", "Vy": "managua", "Vz": "bwr_r",
        }
        
        latex_map = {
            "T": r"$T$", "P": r"$P$", "rho": r"$\rho$",
            "Vx": r"$v_x$", "Vy": r"$v_y$", "Vz": r"$v_z$",
            "Bx": r"$B_x$", "By": r"$B_y$", "Bz": r"$B_z$",
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in self.data:
                print(f"  Skipping {q}: not in data")
                continue
            
            arr = self.data[q]
            q_unit = arr.unit.to_string("latex") if hasattr(arr, 'unit') else ""
            arr = arr.value if hasattr(arr, 'value') else arr
            if arr.ndim != 3 or not (0 <= x_idx < arr.shape[0]):
                print(f"  Skipping {q}: invalid shape or x_idx")
                continue
            
            slice_q = arr[x_idx, :, :]
            
            display_slice = slice_q.copy()
            q_label = latex_map.get(q, q)
            
            if q in ["rho", "P"]:
                display_slice = np.log10(np.clip(display_slice, 1e-40, None))
                q_label = q_label + " (log10)"
            
            cmap_name = cmap_map.get(q, default_cmap)
            cmap = plt.get_cmap(cmap_name) if cmap_name in plt.colormaps() else plt.get_cmap(default_cmap)
            
            if q in ["Bx", "By", "Bz", "Vz", "Vx", "Vy"]:
                vmin, vmax = np.quantile(display_slice, [0.01, 0.99])
                maxabs = max(abs(vmin), abs(vmax))
                vmin, vmax = -maxabs, maxabs
            else:
                vmin, vmax = None, None
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                fig, ax = plt.subplots(figsize=figsize)
                im = ax.imshow(display_slice.T, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
                
                ax.set_xticks([])
                
                scalebar = ScaleBar(pixel_size_km, 'km', length_fraction=0.25, location='lower left', pad=0.5)
                ax.add_artist(scalebar)
                
                # Overlay logtau = 0 contour
                if self.logtau is not None and self.logtau.shape == arr.shape:
                    tau_slice = self.logtau[x_idx, :, :]
                    ys = np.arange(slice_q.shape[0])
                    zs = np.arange(slice_q.shape[1])
                    CS = ax.contour(ys, zs, tau_slice.T, levels=[0.0], colors='white', linewidths=1.5)
                    
                    if len(CS.allsegs[0]) != 0:
                        segs = CS.allsegs[0]
                        longest = max(segs, key=lambda s: s.shape[0])
                        mid = longest[len(longest) // 2]
                        x_lab, y_lab = mid[0], mid[1]
                        ax.text(x_lab + 1, y_lab + 10, r"$\log \tau = 0$",
                                color="white", fontsize=10, ha="left", va="bottom",
                                bbox=dict(facecolor="black", alpha=0.4, boxstyle="round,pad=0.2"))
                
                # Reference layer marker
                if self.z0 is not None and 0 <= self.z0 < slice_q.shape[1]:
                    ax.axhline(self.z0, color='yellow', lw=1.5, ls='--')
                    ax.text(2, self.z0 + 2, r'$T\approx 5780$', color='yellow', fontsize=9,
                            bbox=dict(facecolor='black', alpha=0.4, boxstyle='round,pad=0.2'))
                
                # Height ticks
                n_z = slice_q.shape[1]
                n_ticks = 7
                tick_positions = np.linspace(0, n_z - 1, n_ticks)
                
                if self.height_levels is not None and len(self.height_levels) == n_z:
                    tick_labels = [f"{int(self.height_levels[int(p)])}" for p in tick_positions]
                else:
                    tick_labels = [f"{int((p - self.z0) * dz_km):+d}" for p in tick_positions]
                
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels)
                ax.set_ylabel("Height (km)")
                
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"{q_unit}")
                ax.set_title(f"{q_label} slice at x={x_idx}")
                plt.tight_layout()
                
                if return_figs:
                    figs.append((q, fig))
                else:
                    if save_path is not None:
                        fname = f"{q}_slice_x{x_idx}.png"
                        fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                        print(f"Saved: {save_path / fname}")
                        plt.close(fig)
                    else:
                        plt.show()
                        plt.close(fig)
        
        if return_figs:
            return figs
    
    def plot_medians_vs_logtau(self, new_logtau: np.ndarray,
                                quantities: Optional[List[str]] = None,
                                figsize: Tuple[int, int] = (5, 4),
                                save_dir: Optional[str] = None,
                                dpi: int = 150,
                                return_figs: bool = False):
        """Plot the horizontal median of each remapped quantity against log tau.

        Args:
            new_logtau: The log tau grid used in :meth:`remap_to_optical_depth`.
            quantities: Keys of ``self.od_data`` to plot. Default: all.
            figsize: Figure size in inches.
            save_dir: Save PNGs there instead of showing them.
            dpi: Resolution of saved figures.
            return_figs: Return the figures instead of showing or saving them.

        Returns:
            A list of ``(key, Figure)`` if ``return_figs``; otherwise ``None``.

        Raises:
            ValueError: If :meth:`remap_to_optical_depth` has not been run.

        Examples:
            >>> figs = mhd.plot_medians_vs_logtau(new_logtau, ["T"], return_figs=True)  # doctest: +SKIP
        """
        if not self.od_data:
            raise ValueError("No optical depth data available. Call remap_to_optical_depth() first.")
        
        if quantities is None:
            quantities = list(self.od_data.keys())
        
        latex_map = {
            "T": r"$T$",
            "P": r"$P$",
            "rho": r"$\rho$",
            "Bx": r"$B_x$",
            "By": r"$B_y$",
            "Bz": r"$B_z$",
            "Vx": r"$v_x$",
            "Vy": r"$v_y$",
            "Vz": r"$v_z$"
        }
        
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        else:
            save_path = None
        
        figs = [] if return_figs else None
        
        for q in quantities:
            if q not in self.od_data:
                print(f"  Skipping {q}: not in optical depth data")
                continue
            
            data_q = self.od_data[q]
            q_unit = data_q.unit.to_string("latex") if hasattr(data_q, 'unit') else ""
            median_values = np.median(data_q, axis=(0, 1))
            
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(new_logtau, median_values, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel(r'$\log \tau$', fontsize=12)
            ax.set_ylabel(f'Median {latex_map.get(q, q)} ({q_unit})', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Median {latex_map.get(q, q)} vs Optical Depth', fontsize=12)
            plt.tight_layout()
            
            if return_figs:
                figs.append((q, fig))
            else:
                if save_path is not None:
                    fname = f"{q}_median_vs_logtau.png"
                    fig.savefig(save_path / fname, dpi=dpi, bbox_inches='tight')
                    print(f"Saved: {save_path / fname}")
                    plt.close(fig)
                else:
                    plt.show()
                    plt.close(fig)
        
        if return_figs:
            return figs


# ============================================================================
# EXAMPLE USAGE (MHD)
# ============================================================================
"""
from pathlib import Path
import numpy as np

# Initialize
mhd = MhdData(
    data_path="/path/to/muram-simulation/",
    nx=480, ny=480, nz=256
)

# Load a simulation step
mhd.load_step(step=223, z_max=250)

# Load opacity table and compute optical depth
mhd.load_opacity_table(kappa_path="../csv/kappa.0.dat")
mhd.compute_optical_depth(dz=1e6)

# Remap to optical depth coordinates
new_logtau = np.arange(-2.0, 0.1, 0.1)
mhd.remap_to_optical_depth(new_logtau, quantities=["T", "Vz", "Bz"])

# Plot geometric height surfaces
mhd.plot_surfaces(
    z_idx=190,
    data_source="geometric",
    save_dir="./output/surfaces"
)

# Plot vertical slices with optical depth contours
mhd.plot_slices(
    x_idx=120,
    save_dir="./output/slices"
)

# Plot optical depth surfaces
limit_values = {
    "T": (5500, 7000),
    "Vz": (-5, 5),
    "Bz": (-100, 100),
}
mhd.plot_surfaces(
    z_idx=20,
    data_source="optical_depth",
    quantities=["T", "Vz", "Bz"],
    limit_values=limit_values,
    save_dir="./output/surfaces_od"
)

# Plot median values vs optical depth
mhd.plot_medians_vs_logtau(
    new_logtau=new_logtau,
    quantities=["T", "Bz", "Vz"],
    save_dir="./output/medians"
)

# Access processed data
T_data = mhd.data["T"]  # Geometric height data
T_od = mhd.od_data["T"]  # Optical depth data
logtau = mhd.logtau  # Optical depth map
"""

# ============================================================================
# StokesData class
# ============================================================================
class StokesData:
    """Synthetic Stokes profiles of one MURaM step and their degradation to Hinode/SP.

    ``self.data`` maps ``"I"``, ``"Q"``, ``"U"``, ``"V"`` to ``(nx, ny, n_wl)`` arrays on
    ``self.wl`` (Angstrom). The training pipeline applies, in order,
    :meth:`load_hinode_lsf`, :meth:`apply_spectral_convolution` and
    :meth:`resample_to_hinode`, which leaves 112 points on the Hinode grid.
    :meth:`add_hinode_noise` exists but is not called by training yet.

    Attributes:
        wl: Current wavelength axis in Angstrom (synthesis grid, then the Hinode grid).
        mean_continuum: Continuum map ``(nx, ny)`` in units of the reference continuum.
        circular_polarization: Map of mean |V| / I over wavelength (dimensionless).
        lsf_kernel: Spectral LSF sampled at the synthesis step, normalized to sum 1.
    """
    
    def __init__(self, data_dir: str, step: Union[int, str], wavelength_range: Tuple[float, float] = (6300.5, 6303.5), 
                 wavelength_step: float = 0.01):
        """Set up an empty container for one step.

        Args:
            data_dir: Folder with ``stokes_<step>.npy`` (legacy source).
            step: Step number; see :meth:`_normalize_step`.
            wavelength_range: First and end wavelength of the synthesis grid in Angstrom
                (the end is excluded, as in ``np.arange``).
            wavelength_step: Synthesis sampling in Angstrom (10 mA for the NICOLE data).
        """
        self.data_dir = Path(data_dir)
        self.step = self._normalize_step(step)
        self.wl = np.arange(wavelength_range[0], wavelength_range[1], wavelength_step)
        self.data: Dict[str, np.ndarray] = {}
        self.mean_continuum: Optional[np.ndarray] = None
        self.circular_polarization: Optional[np.ndarray] = None
        self.hinode_wl: Optional[np.ndarray] = None
        self.lsf_kernel: Optional[np.ndarray] = None
        self.nx: Optional[int] = None
        self.ny: Optional[int] = None
        self.nwl: Optional[int] = None

    def _normalize_step(self, step: Union[int, str]) -> str:
        """Return the six-digit file suffix of a step: ``110`` or ``"110"`` gives ``"110000"``."""
        if isinstance(step, int):
            return f"{step:03d}000"
        else:
            s = str(step)
            if s.isdigit() and len(s) == 3:
                return f"{int(s):03d}000"
            elif s.isdigit() and len(s) == 6:
                return s
            else:
                return s

    def load_stokes(self) -> Dict[str, np.ndarray]:
        """Load the legacy ``stokes_<step>.npy`` cube into I, Q, U, V (legacy source only).

        The file is ``(nx, ny, n_wl, 4)``. Each map is rotated by 90 degrees and flipped
        to match the orientation of the MHD arrays.

        Returns:
            ``self.data`` with ``(nx, ny, n_wl)`` arrays for I, Q, U and V.

        Raises:
            FileNotFoundError: If the file does not exist.

        Examples:
            >>> s = StokesData("data/muram-simulation/", 110)  # doctest: +SKIP
            >>> s.load_stokes()["I"].shape                       # doctest: +SKIP
            (480, 480, 300)
        """
        stokes_path = self.data_dir / f"stokes_{self.step}.npy"
        if not stokes_path.exists():
            raise FileNotFoundError(f"Stokes file not found: {stokes_path}")
        print(f"Loading Stokes data from {stokes_path}")
        stokes_cube = np.load(stokes_path)
        self.data = {
            "I": stokes_cube[:, :, :, 0],
            "Q": stokes_cube[:, :, :, 1],
            "U": stokes_cube[:, :, :, 2],
            "V": stokes_cube[:, :, :, 3]
        }
        for key in self.data.keys():
            self.data[key] = np.flip(np.rot90(self.data[key], k=1, axes=(0,1)), axis=0)
        self.nx, self.ny, self.nwl = self.data["I"].shape
        for key in self.data.keys():
            print(f"  {key} shape: {self.data[key].shape}")
        print("  Done.")
        return self.data

    def continuum_normalization(
        self,
        cont_indices: Optional[List[int]] = None,
        fixed_ic: Optional[float] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Divide all four Stokes parameters by one continuum level and store the continuum map.

        Args:
            cont_indices: Wavelength indices treated as continuum. Default ``[0, 1, 2, 3]``.
            fixed_ic: Continuum level to divide by, in the file's intensity units. ``None``
                uses this step's spatial mean over ``cont_indices``, so the step's mean
                continuum becomes exactly 1.

        Returns:
            ``(self.data, mean_continuum)``, where ``mean_continuum`` is the ``(nx, ny)``
            continuum map after normalization.

        Raises:
            ValueError: If no data are loaded, ``cont_indices`` is empty or not 1-D, or
                ``fixed_ic`` is not finite and positive.

        Examples:
            >>> import contextlib, io
            >>> s = StokesData(".", 110)
            >>> s.data = {"I": np.full((2, 2, 5), 4.0), "V": np.full((2, 2, 5), 0.4)}
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     data, cont = s.continuum_normalization()
            >>> float(cont.mean()), float(data["V"][0, 0, 0])
            (1.0, 0.1)
        """
        if not self.data:
            raise ValueError("Stokes data not loaded. Call load_stokes() first.")
        if cont_indices is None:
            cont_indices = [0, 1, 2, 3]
        cont_indices = np.asarray(cont_indices, dtype=int)
        if cont_indices.ndim != 1 or cont_indices.size == 0:
            raise ValueError("cont_indices must be a 1D non-empty array-like of indices")

        print("Applying continuum normalization...")
        if fixed_ic is None:
            I_c = float(self.data["I"][:, :, cont_indices].mean(axis=2).mean())
            ic_mode = "per-step"
        else:
            I_c = float(fixed_ic)
            if not np.isfinite(I_c) or I_c <= 0:
                raise ValueError(f"fixed_ic must be finite and > 0, got {fixed_ic}")
            ic_mode = "fixed-global"

        for key in self.data.keys():
            self.data[key] = self.data[key] / I_c
        mean_continuum = self.data["I"][:, :, cont_indices].mean(axis=2)
        self.mean_continuum = mean_continuum
        print(f"  Continuum mode: {ic_mode}")
        print(f"  Global continuum used: I_c = {I_c:.6e}")
        print("  Done.")
        return self.data, mean_continuum

    def spectropolarimetry(self, stokes_data: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Map of circular polarization, the wavelength mean of |V| / I (dimensionless).

        I is floored at 1e-10 to avoid division by zero.

        Args:
            stokes_data: Dict with ``"I"`` and ``"V"``. Default ``self.data``.

        Returns:
            ``(nx, ny)`` array, also stored in ``self.circular_polarization``.

        Raises:
            ValueError: If I or V is missing.

        Examples:
            >>> s = StokesData(".", 110)
            >>> I = np.ones((1, 1, 4)); V = np.array([[[0.02, -0.02, 0.0, 0.0]]])
            >>> s.spectropolarimetry({"I": I, "V": V}).tolist()
            [[0.01]]
        """
        if stokes_data is None:
            stokes_data = self.data
        if "V" not in stokes_data or "I" not in stokes_data:
            raise ValueError("Stokes V and I required for circular polarization")
        nwl_points = stokes_data["I"].shape[-1]
        epsilon = 1e-10
        I_safe = np.maximum(stokes_data["I"], epsilon)
        circular_pol = np.sum(np.abs(stokes_data["V"]) / I_safe, axis=-1) / nwl_points
        if np.isnan(circular_pol).any() or np.isinf(circular_pol).any():
            print("  WARNING: NaN or Inf values in circular polarization")
        self.circular_polarization = circular_pol
        return circular_pol

    def rms_calculation(self, data: np.ndarray) -> float:
        """Relative RMS (contrast) of a map: std / mean, dimensionless.

        Args:
            data: Any array, e.g. the continuum map.

        Returns:
            The population standard deviation divided by the mean.

        Examples:
            >>> StokesData(".", 110).rms_calculation(np.array([0.9, 1.1]))
            0.1...
        """
        return np.sqrt(np.mean((data - data.mean())**2)) / data.mean()

    def load_hinode_lsf(self, lsf_path: str) -> np.ndarray:
        """Read the Hinode/SP spectral LSF and resample it to the synthesis step.

        The file has two columns, offset from line centre (A) and response. The response is
        normalized, interpolated with a cubic spline on the synthesis step between the file's
        ends minus 10 mA, and normalized again to sum 1. With the shipped file and a 10 mA
        step the kernel has 13 points and its centroid sits 0.0075 samples from the centre,
        i.e. it shifts a line by about 0.004 km/s.

        Args:
            lsf_path: Path to ``hinode_sp.spline.psf``.

        Returns:
            The kernel, also stored in ``self.lsf_kernel``.

        Examples:
            >>> s = StokesData(".", 110)
            >>> k = s.load_hinode_lsf("data/hinode-MODEST/PSFs/hinode_sp.spline.psf")  # doctest: +SKIP
            >>> k.size, round(float(k.sum()), 6)  # doctest: +SKIP
            (13, 1.0)
        """
        print(f"Loading Hinode LSF from {lsf_path}")
        lsf_data = pd.read_csv(lsf_path, sep=r"\s+", header=None)
        lsf_wl_orig = lsf_data[0].values
        lsf_values_orig = lsf_data[1].values / lsf_data[1].values.sum()
        lsf_interp = interp1d(lsf_wl_orig, lsf_values_orig, kind="cubic")
        lsf_wl_min = np.round(lsf_wl_orig.min() + 0.010, 3)
        lsf_wl_max = np.round(lsf_wl_orig.max() - 0.010, 3)
        step = self.wl[1] - self.wl[0]
        new_kernel_wl = np.arange(lsf_wl_min, lsf_wl_max + step, step)
        new_kernel_values = lsf_interp(new_kernel_wl)
        new_kernel_values = new_kernel_values / new_kernel_values.sum()
        self.lsf_kernel = new_kernel_values
        print(f"  LSF wavelength range: {lsf_wl_min:.3f} - {lsf_wl_max:.3f} Å")
        print(f"  LSF kernel size: {len(self.lsf_kernel)}")
        print("  Done.")
        return self.lsf_kernel

    def apply_spectral_convolution(self) -> Dict[str, np.ndarray]:
        """Convolve every Stokes parameter with the LSF along wavelength.

        Uses ``scipy.ndimage.convolve1d`` with ``mode="wrap"``: the spectrum is treated as
        periodic, so the first and last few samples mix with each other. This is harmless
        when both ends of the synthesis window are continuum.

        Returns:
            ``self.data`` with the convolved arrays (same shapes).

        Raises:
            ValueError: If no data or no kernel is loaded.

        Examples:
            >>> import contextlib, io
            >>> s = StokesData(".", 110)
            >>> s.data = {"I": np.ones((1, 1, 8))}
            >>> s.lsf_kernel = np.array([0.25, 0.5, 0.25])
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     out = s.apply_spectral_convolution()
            >>> out["I"].ravel().tolist() == [1.0] * 8   # a flat spectrum stays flat
            True
        """
        if not self.data:
            raise ValueError("Stokes data not loaded. Call load_stokes() first.")
        if self.lsf_kernel is None:
            raise ValueError("LSF kernel not loaded. Call load_hinode_lsf() first.")
        print("Applying spectral convolution with LSF...")
        for key in self.data.keys():
            self.data[key] = convolve1d(
                self.data[key],
                weights=self.lsf_kernel,
                axis=-1,
                mode='wrap'
            )
            print(f"  {key}: convolved")
        print("  Done.")
        return self.data

    def resample_to_hinode(self, hinode_wl: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Resample the synthesized profiles onto the Hinode/SOT-SP spectral grid (112 points).

        The grid defaults to :func:`utils.hinode_wavelengths.hinode_wavelength_grid`, which is
        derived from the MODEST FITS header, the same axis the real observations are loaded
        on. Interpolation is cubic along wavelength.

        This used to build the grid from a hardcoded CRVAL1=6302.0, CDELT1=0.0215,
        CRPIX1=57 triple, which gave [6300.7960, 6303.1825] A while the observations are on
        [6300.8736, 6303.2576] A. That 0.0776 A disagreement (about 3.6 spectral pixels)
        made every model trained here read a spurious 3.7 km/s blueshift on MODEST data.

        Args:
            hinode_wl: Explicit target grid in Angstrom. It must equal the FITS-derived grid
                to 1e-9 A; the argument exists only for callers that already hold it.

        Returns:
            ``self.data`` with ``(nx, ny, 112)`` arrays; ``self.wl`` becomes the Hinode grid.

        Raises:
            ValueError: If no data are loaded.
            AssertionError: If ``hinode_wl`` differs from the FITS-derived grid, or if the
                synthesis grid does not cover it (the cubic interpolation would extrapolate).

        Examples:
            >>> s.resample_to_hinode()["I"].shape  # doctest: +SKIP
            (480, 480, 112)
        """
        if not self.data:
            raise ValueError("No Stokes data available. Call load_stokes() first.")
        if hinode_wl is None:
            hinode_wl = hinode_wavelength_grid()
        self.hinode_wl = np.asarray(hinode_wl, dtype=np.float64)
        NAXIS1 = int(self.hinode_wl.size)

        # Guard the invariant that was silently violated before: the axis training
        # data is resampled onto MUST be the FITS-derived axis the observations are
        # loaded on. Any reintroduced hardcoded grid fails loudly here instead of
        # quietly biasing V_LOS by ~3.7 km/s.
        canonical = hinode_wavelength_grid(NAXIS1)
        assert np.allclose(self.hinode_wl, canonical, atol=1e-9), (
            "Hinode resample grid does not match the MODEST FITS-derived axis.\n"
            f"  got       : [{self.hinode_wl[0]:.4f}, {self.hinode_wl[-1]:.4f}] ({NAXIS1} pts)\n"
            f"  canonical : [{canonical[0]:.4f}, {canonical[-1]:.4f}] ({canonical.size} pts)\n"
            f"  max offset: {np.max(np.abs(self.hinode_wl - canonical)):.4e} A "
            f"(~{3e5 * np.max(np.abs(self.hinode_wl - canonical)) / 6302.0:.3f} km/s of spurious Doppler shift)\n"
            "Get the axis from utils.hinode_wavelengths, never from hardcoded "
            "CRVAL1/CDELT1/CRPIX1 values."
        )
        # The synthesis grid must bracket the target grid, or the cubic interpolation
        # below would extrapolate (silently, for out-of-range points).
        src = np.asarray(self.wl, dtype=np.float64)
        assert src[0] <= self.hinode_wl[0] and src[-1] >= self.hinode_wl[-1], (
            f"Synthesis grid [{src[0]:.4f}, {src[-1]:.4f}] A does not cover the target "
            f"Hinode grid [{self.hinode_wl[0]:.4f}, {self.hinode_wl[-1]:.4f}] A; "
            "resampling would extrapolate. Widen the synthesis wavelength range."
        )

        print(f"Resampling to Hinode/SP wavelengths...")
        print(f"  {NAXIS1} wavelength points from {self.hinode_wl[0]:.4f} to {self.hinode_wl[-1]:.4f} Å")
        for key in self.data.keys():
            stokes_reshaped = self.data[key].reshape(-1, self.data[key].shape[2])
            interp_func = interp1d(self.wl, stokes_reshaped, kind="cubic", axis=1)
            stokes_interp = interp_func(self.hinode_wl).reshape(self.nx, self.ny, NAXIS1)
            self.data[key] = stokes_interp
            print(f"  {key}: resampled to shape {stokes_interp.shape}")
        self.wl = self.hinode_wl
        self.nwl = NAXIS1
        print("  Done.")
        return self.data

    def add_hinode_noise(self, noise_level: float = 5.9e-4, 
                        continuum_intensity: Optional[float] = None) -> Dict[str, np.ndarray]:
        """Add white Gaussian noise with sigma = ``noise_level`` times the continuum to all parameters.

        The same sigma is used for I, Q, U and V, and a new realization is drawn on every
        call (``np.random``, not seeded). Training does not call this yet.

        Args:
            noise_level: Noise as a fraction of the continuum (5.9e-4 by default).
            continuum_intensity: Continuum level in the data's units. Default: the mean of I
                over all pixels and wavelengths.

        Returns:
            ``self.data`` with noise added.

        Raises:
            ValueError: If no data are loaded.

        Examples:
            >>> import contextlib, io
            >>> s = StokesData(".", 110)
            >>> s.data = {"V": np.zeros((50, 50, 20))}
            >>> s.nx, s.ny, s.nwl = 50, 50, 20
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = s.add_hinode_noise(1e-3, continuum_intensity=1.0)
            >>> bool(abs(s.data["V"].std() - 1e-3) < 5e-5)
            True
        """
        if not self.data:
            raise ValueError("Stokes data not available. Call resample_to_hinode() first.")
        if continuum_intensity is None:
            continuum_intensity = self.data["I"].flatten().mean()
        sigma = noise_level * continuum_intensity
        print(f"Adding Hinode noise...")
        print(f"  Continuum intensity: {continuum_intensity:.6e}")
        print(f"  Noise level: σ = {sigma:.6e}")
        for key in self.data.keys():
            noise = sigma * np.random.randn(self.nx, self.ny, self.nwl)
            self.data[key] = self.data[key] + noise
            print(f"  {key}: noise added")
        print("  Done.")
        return self.data

    def plot_selected_pixel(self, x: int, y: int, 
                           save_dir: Optional[str] = None, dpi: int = 300) -> None:
        """Show the map of I at the first wavelength with a box around pixel (x, y).

        Args:
            x: Pixel index along the first axis.
            y: Pixel index along the second axis.
            save_dir: Save ``selected_pixel.png`` there instead of showing it.
            dpi: Resolution of the saved figure.

        Raises:
            ValueError: If I is not loaded.

        Examples:
            >>> s.plot_selected_pixel(100, 200, save_dir="figs")  # doctest: +SKIP
        """
        if "I" not in self.data:
            raise ValueError("Stokes I not loaded. Call load_stokes() first.")
        fig, ax = plt.subplots(figsize=(5, 5))
        data = self.data["I"][..., 0]
        im = ax.imshow(data, cmap='inferno', origin="lower")
        rect = patches.Rectangle((y - 1.5, x - 1.5), 3*4, 3*4, linewidth=3, edgecolor='yellow', facecolor='none')
        ax.add_patch(rect)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im, cax=cax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Intensity with Highlighted Pixel")
        plt.tight_layout()
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path / "selected_pixel.png", dpi=dpi)
            print(f"Saved: {save_path / 'selected_pixel.png'}")
        else:
            plt.show()
        plt.close(fig)

    def plot_stokes(self, ax, x: int, y: int,
                    wl_index_range: Optional[Tuple[int, int]] = None,
                    scatter: bool = False,
                    **kwargs):
        """Plot I and V of pixel (x, y) on two given axes.

        Args:
            ax: Sequence of two Matplotlib axes, for I and V.
            x: Pixel index along the first axis.
            y: Pixel index along the second axis.
            wl_index_range: ``(start, stop)`` wavelength indices to plot. Default: all.
            scatter: Draw points instead of lines.
            **kwargs: Passed to ``plot`` or ``scatter`` (e.g. ``label``).

        Returns:
            The axes.

        Raises:
            ValueError: If I or V is not loaded.

        Examples:
            >>> fig, ax = plt.subplots(1, 2)                  # doctest: +SKIP
            >>> s.plot_stokes(ax, 100, 200, label="MURaM")    # doctest: +SKIP
        """
        if "I" not in self.data or "V" not in self.data:
            raise ValueError("Stokes I and V must be loaded. Call load_stokes() first.")

        stokes_titles = [r"$I$", r"$V/I_c$"]
        for i, key in enumerate(["I", "V"]):
            wl_to_plot = self.wl
            if wl_index_range is not None:
                wl_to_plot = self.wl[wl_index_range[0]:wl_index_range[1]]
                stokes_to_plot = self.data[key][x, y, wl_index_range[0]:wl_index_range[1]]
            else:
                stokes_to_plot = self.data[key][x, y, :]

            if scatter:
                ax[i].scatter(wl_to_plot, stokes_to_plot, **kwargs)
            else:
                ax[i].plot(wl_to_plot, stokes_to_plot, **kwargs)

            ax[i].set_title(stokes_titles[i])
            ax[i].set_xlabel("Wavelength (Å)")
            ax[i].legend()

        return ax

    def plot_polarizations(
        self,
        continuum_map: Optional[np.ndarray] = None,
        polarization_map: Optional[np.ndarray] = None,
        titles: Tuple[str, str] = ("Continuum", "Circular Pol."),
        rms_values: Optional[Tuple[float, float]] = None,
        quantiles: Tuple[float, float] = (0.05, 0.95),
        save_dir: Optional[str] = None,
        dpi: int = 150,
    ) -> None:
        """Plot the continuum and circular-polarization maps side by side, with their relative RMS.

        Args:
            continuum_map: ``(nx, ny)`` map. Default ``self.mean_continuum``.
            polarization_map: ``(nx, ny)`` map. Default ``self.circular_polarization``.
            titles: Panel titles.
            rms_values: Values printed in the titles. Default: :meth:`rms_calculation` of
                each map.
            quantiles: Lower and upper quantiles that set the polarization colour range.
            save_dir: Save ``polarizations.png`` there instead of showing it.
            dpi: Resolution of the saved figure.

        Raises:
            ValueError: If either map is unavailable.

        Examples:
            >>> s.plot_polarizations(save_dir="figs")  # doctest: +SKIP
        """
        continuum = continuum_map if continuum_map is not None else self.mean_continuum
        polarization = polarization_map if polarization_map is not None else self.circular_polarization

        if continuum is None or polarization is None:
            raise ValueError("Continuum and polarization data must be available to plot.")

        if rms_values is None:
            rms_values = (
                self.rms_calculation(continuum),
                self.rms_calculation(polarization),
            )

        q_low, q_high = np.quantile(polarization, quantiles)

        fig, ax = plt.subplots(1, 2, figsize=(8.6, 4.3))

        im1 = ax[0].imshow(continuum, cmap="inferno", origin="lower")
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        divider1 = make_axes_locatable(ax[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1)

        im2 = ax[1].imshow(polarization, cmap="PiYG", vmin=q_low, vmax=q_high, origin="lower")
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        divider2 = make_axes_locatable(ax[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im2, cax=cax2)

        ax[0].set_title(f"{titles[0]} (rms = {rms_values[0]:.2f})")
        ax[1].set_title(f"{titles[1]} (rms = {rms_values[1]:.2f})")

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            outfile = save_path / "polarizations.png"
            fig.savefig(outfile, dpi=dpi)
            print(f"Saved: {outfile}")
            plt.close(fig)
        else:
            plt.show()
            plt.close(fig)


def _otsu_threshold(values: np.ndarray, n_bins: int = 256) -> float:
    """Otsu threshold of the finite values: the cut that maximizes the between-class variance.

    Args:
        values: Any array; NaN and inf are ignored.
        n_bins: Number of histogram bins.

    Returns:
        The threshold, a histogram bin centre. A constant array returns its value.

    Raises:
        ValueError: If there are no finite values.

    Examples:
        >>> x = np.concatenate([np.full(100, 0.8), np.full(100, 1.2)])
        >>> 0.8 < _otsu_threshold(x) < 1.2
        True
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("Cannot compute Otsu threshold on empty/non-finite array.")
    if np.allclose(arr.min(), arr.max()):
        return float(arr.min())

    hist, bin_edges = np.histogram(arr, bins=int(n_bins))
    hist = hist.astype(np.float64)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    prob = hist / (hist.sum() + 1e-12)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)
    sigma_b2[~np.isfinite(sigma_b2)] = -np.inf
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def build_granulation_polarization_masks(
    mean_continuum: np.ndarray,
    circular_polarization: np.ndarray,
    continuum_threshold: Optional[float] = None,
    polarization_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Split pixels into four regions: granular or intergranular, crossed with strong or weak polarization.

    A pixel is granular if its continuum is above the threshold, and weakly polarized if
    its circular polarization is below the threshold. Thresholds default to the Otsu
    threshold of each map.

    Args:
        mean_continuum: Continuum map ``(nx, ny)``, normalized intensity.
        circular_polarization: Map of mean |V| / I, same shape.
        continuum_threshold: Fixed continuum threshold instead of Otsu.
        polarization_threshold: Fixed polarization threshold instead of Otsu.

    Returns:
        Dict with ``continuum_threshold``, ``polarization_threshold``, ``masks`` (four
        boolean maps: ``granular_strong``, ``intergranular_strong``, ``granular_weak``,
        ``intergranular_weak``) and ``counts`` (pixels per mask).

    Raises:
        ValueError: If the two maps have different shapes.

    Examples:
        >>> cont = np.array([[1.1, 0.9], [1.1, 0.9]])
        >>> pol = np.array([[0.01, 0.01], [0.10, 0.10]])
        >>> out = build_granulation_polarization_masks(cont, pol, 1.0, 0.05)
        >>> out["counts"]
        {'granular_strong': 1, 'intergranular_strong': 1, 'granular_weak': 1, 'intergranular_weak': 1}
    """
    mean_continuum = np.asarray(mean_continuum)
    circular_polarization = np.asarray(circular_polarization)

    if mean_continuum.shape != circular_polarization.shape:
        raise ValueError(
            "mean_continuum and circular_polarization must have the same shape. "
            f"Got {mean_continuum.shape} vs {circular_polarization.shape}."
        )

    t_cont = float(continuum_threshold) if continuum_threshold is not None else _otsu_threshold(mean_continuum)
    t_pol = float(polarization_threshold) if polarization_threshold is not None else _otsu_threshold(circular_polarization)

    granular = mean_continuum > t_cont
    intergranular = ~granular
    weak_pol = circular_polarization < t_pol
    strong_pol = ~weak_pol

    masks = {
        "granular_strong": granular & strong_pol,
        "intergranular_strong": intergranular & strong_pol,
        "granular_weak": granular & weak_pol,
        "intergranular_weak": intergranular & weak_pol,
    }
    counts = {k: int(v.sum()) for k, v in masks.items()}

    return {
        "continuum_threshold": t_cont,
        "polarization_threshold": t_pol,
        "masks": masks,
        "counts": counts,
    }


def build_balanced_region_indices(
    region_masks: Dict[str, np.ndarray],
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Dict[str, int]]]:
    """Draw the same number of pixels from each of the four region masks.

    Each region contributes as many pixels as the smallest one has, drawn without
    replacement; the result is shuffled.

    Args:
        region_masks: The four boolean maps from :func:`build_granulation_polarization_masks`.
        rng: Random generator. Default: a fresh unseeded one.

    Returns:
        ``(indices, info)``: flattened pixel indices (int64) and a dict with
        ``counts_before`` and ``counts_after`` per region.

    Raises:
        KeyError: If a region mask is missing.
        ValueError: If a region is empty.

    Examples:
        >>> masks = {k: np.zeros((3, 3), bool) for k in
        ...          ["granular_strong", "intergranular_strong", "granular_weak", "intergranular_weak"]}
        >>> masks["granular_strong"][0] = True          # 3 pixels
        >>> masks["intergranular_strong"][1, :2] = True  # 2 pixels
        >>> masks["granular_weak"][2, 0] = True          # 1 pixel
        >>> masks["intergranular_weak"][2, 1:] = True    # 2 pixels
        >>> idx, info = build_balanced_region_indices(masks, np.random.default_rng(0))
        >>> idx.size, info["counts_after"]["granular_strong"]
        (4, 1)
    """
    required = [
        "granular_strong",
        "intergranular_strong",
        "granular_weak",
        "intergranular_weak",
    ]
    missing = [k for k in required if k not in region_masks]
    if missing:
        raise KeyError(f"Missing region masks: {missing}")

    if rng is None:
        rng = np.random.default_rng()

    idx_by_region = {name: np.flatnonzero(region_masks[name].ravel()) for name in required}
    counts_before = {name: int(idx.size) for name, idx in idx_by_region.items()}
    target = min(counts_before.values()) if counts_before else 0

    if target <= 0:
        raise ValueError(f"At least one region has zero pixels: {counts_before}")

    selected_chunks = []
    for name in required:
        chosen = rng.choice(idx_by_region[name], size=target, replace=False)
        selected_chunks.append(chosen)

    selected = np.concatenate(selected_chunks)
    rng.shuffle(selected)

    counts_after = {name: int(target) for name in required}
    counts_after["total_selected"] = int(selected.size)
    return selected.astype(np.int64), {
        "counts_before": counts_before,
        "counts_after": counts_after,
    }


def build_bz_balance_bin_edges(
    scores: np.ndarray,
    n_bins: int = 12,
    balance_cap: Optional[float] = None,
    bin_scale: str = "log",
) -> Tuple[np.ndarray, float]:
    """Bin edges for |B_LOS|-strength balancing. Returns ``(edges, cap)``.

    Shared by the per-step and global balancers so the two cannot drift apart.

    ``bin_scale`` matters more than it looks, because MURaM's |Bz| spans four decades with
    about 90 % of pixels under 100 G.

    * ``"linear"``: edges evenly spaced up to the cap. Nearly every pixel lands in bin 0
      and the top bins hold single pixels; this original scheme collapsed the selection
      to about 96 of 691k pixels.
    * ``"quantile"``: every bin holds the same count, so almost every bin sits in the weak
      regime (11 of 13 below 93 G) and the 100 to 900 G range that sunspot crops need is
      squeezed into one bin. Balancing then barely shifts the distribution.
    * ``"log"`` (default): edges spread over the dynamic range, so each decade gets its own
      bin. This moves the 90th percentile the model sees from about 88 G to about 320 G.

    All variants cap the top edge at ``balance_cap`` (default: the 99.9th percentile) and
    put everything above it in the last bin, which ends at the maximum. Edges are rounded
    to float32 and deduplicated.

    Args:
        scores: Per-pixel |B| scores in G.
        n_bins: Requested number of bins below the cap (at least 2).
        balance_cap: Cap in G. ``None`` uses the 99.9th percentile.
        bin_scale: ``"log"``, ``"quantile"`` or ``"linear"``.

    Returns:
        ``(edges, cap)``: increasing float32 edges from the minimum to the maximum score,
        and the cap actually used, in G.

    Examples:
        >>> scores = np.geomspace(1.0, 1000.0, 1000)
        >>> edges, cap = build_bz_balance_bin_edges(scores, n_bins=4, balance_cap=500.0)
        >>> float(edges[0]), cap, float(edges[-1])
        (1.0, 500.0, 1000.0)
        >>> bool(np.all(np.diff(edges) > 0))
        True
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    score_min, score_max = float(np.min(scores)), float(np.max(scores))
    n_bins = int(max(2, n_bins))

    cap = float(np.percentile(scores, 99.9)) if balance_cap is None else float(balance_cap)
    cap = min(max(cap, score_min + np.finfo(np.float32).eps), score_max)

    if bin_scale == "linear":
        inner = np.linspace(score_min, cap, n_bins)
    elif bin_scale == "quantile":
        below = scores[scores <= cap]
        inner = np.quantile(below if below.size else scores, np.linspace(0.0, 1.0, n_bins))
    else:  # log
        positive = scores[scores > 0]
        floor = max(float(np.percentile(positive, 1)), 1e-3) if positive.size else 1e-3
        floor = min(floor, cap / 2.0)
        inner = np.geomspace(floor, cap, n_bins)

    # Deduplicate AFTER narrowing to float32: edges distinct in float64 can collapse once
    # narrowed, leaving a zero-width bin holding a single pixel that oversampling would then
    # replicate thousands of times.
    edges = np.unique(
        np.concatenate([[score_min], inner[inner < cap], [cap], [score_max]]).astype(np.float32)
    )
    if edges.size < 3:
        edges = np.linspace(score_min, score_max, n_bins + 1, dtype=np.float32)
    return edges, cap


def build_bz_strength_balanced_indices(
    mhd_data: Dict[str, np.ndarray],
    base_selected_indices: Optional[np.ndarray] = None,
    n_bins: int = 12,
    score_mode: str = "mean_abs",
    tau_idx: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    oversample: bool = True,
    balance_cap: Optional[float] = None,
    max_oversample_factor: float = 10.0,
    bin_scale: str = "log",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Resample pixels so that each |B_LOS|-strength bin is about equally represented.

    Each pixel gets a score from its |Bz| over the tau nodes (``score_mode``). Scores are
    binned with :func:`build_bz_balance_bin_edges` and every bin is brought to a target
    count: the median occupied-bin count when oversampling, else the smallest count. Bins
    above the target are subsampled without replacement. Bins below it are drawn with
    replacement, but grow at most ``max_oversample_factor`` times, so a handful of strong
    pixels is not replicated thousands of times. The result is shuffled.

    Args:
        mhd_data: Dict with ``"Bz"`` of shape ``(nx, ny, n_tau)``, in G.
        base_selected_indices: Flattened candidate pixels (e.g. from region balancing).
            Default: all pixels.
        n_bins: Requested number of bins.
        score_mode: ``"mean_abs"`` (mean |Bz| over nodes), ``"max_abs"`` or
            ``"tau_index"`` (|Bz| at node ``tau_idx``).
        tau_idx: Node used by ``"tau_index"``. Default: the middle node.
        rng: Random generator. Default: a fresh unseeded one.
        oversample: Equalize upward (default) instead of down to the rarest bin.
        balance_cap: |Bz| in G above which all pixels share the top bin. Default: the
            99.9th percentile of the candidate scores.
        max_oversample_factor: Largest growth factor of a bin drawn with replacement.
        bin_scale: Passed to :func:`build_bz_balance_bin_edges`.

    Returns:
        ``(indices, info)``: flattened pixel indices (int64, may repeat) and a dict with
        the bin edges, counts before and after, the target and the settings used.

    Raises:
        KeyError: If ``"Bz"`` is missing.
        ValueError: For a non 3-D Bz, an unknown ``score_mode``, an out-of-range
            ``tau_idx`` or candidate index, or no finite scores.

    Examples:
        >>> rng = np.random.default_rng(0)
        >>> bz = np.concatenate([np.full(990, 10.0), np.full(10, 1000.0)])  # 1 % strong pixels
        >>> bz = (bz + rng.random(1000)).reshape(10, 100, 1)
        >>> idx, info = build_bz_strength_balanced_indices({"Bz": bz}, n_bins=2, rng=rng,
        ...                                                 balance_cap=500.0, bin_scale="linear")
        >>> info["target_per_bin"]     # median of the two bin counts, (990 + 10) / 2
        500
        >>> strong = np.abs(bz.reshape(-1)[idx]) > 500
        >>> int(strong.sum()), int((~strong).sum())   # 10 strong pixels grow at most 10x
        (100, 500)
    """
    if "Bz" not in mhd_data:
        raise KeyError("mhd_data must contain a 'Bz' array")

    if rng is None:
        rng = np.random.default_rng()

    bz = np.asarray(mhd_data["Bz"], dtype=np.float32)
    if bz.ndim != 3:
        raise ValueError(f"Expected Bz to be 3D (nx, ny, n_tau), got shape {bz.shape}")

    valid_score_modes = {"mean_abs", "max_abs", "tau_index"}
    if score_mode not in valid_score_modes:
        raise ValueError(f"score_mode must be one of {sorted(valid_score_modes)}, got {score_mode!r}")

    if score_mode == "mean_abs":
        pixel_scores = np.mean(np.abs(bz), axis=2)
    elif score_mode == "max_abs":
        pixel_scores = np.max(np.abs(bz), axis=2)
    else:
        if tau_idx is None:
            tau_idx = bz.shape[2] // 2
        if tau_idx < 0 or tau_idx >= bz.shape[2]:
            raise ValueError(f"tau_idx must be within [0, {bz.shape[2] - 1}], got {tau_idx}")
        pixel_scores = np.abs(bz[:, :, int(tau_idx)])

    score_flat = np.asarray(pixel_scores, dtype=np.float32).ravel()
    if base_selected_indices is None:
        candidate_indices = np.arange(score_flat.size, dtype=np.int64)
    else:
        candidate_indices = np.asarray(base_selected_indices, dtype=np.int64).ravel()

    if candidate_indices.size == 0:
        raise ValueError("base_selected_indices is empty")
    if candidate_indices.min() < 0 or candidate_indices.max() >= score_flat.size:
        raise ValueError("base_selected_indices contains out-of-range values")

    candidate_scores = score_flat[candidate_indices]
    finite_mask = np.isfinite(candidate_scores)
    candidate_indices = candidate_indices[finite_mask]
    candidate_scores = candidate_scores[finite_mask]

    if candidate_indices.size == 0:
        raise ValueError("No finite Bz scores available for balancing")

    score_min = float(np.min(candidate_scores))
    score_max = float(np.max(candidate_scores))
    if np.isclose(score_min, score_max):
        counts_before = {"bin_0": int(candidate_indices.size)}
        return candidate_indices.astype(np.int64), {
            "score_mode": score_mode,
            "tau_idx": None if tau_idx is None else int(tau_idx),
            "n_bins": 1,
            "score_min": score_min,
            "score_max": score_max,
            "bin_edges": [score_min, score_max],
            "counts_before": counts_before,
            "counts_after": {"bin_0": int(candidate_indices.size), "total_selected": int(candidate_indices.size)},
            "total_candidates": int(candidate_indices.size),
        }

    # Bin edges. Equal-width edges up to score_max collapse on MURaM's |Bz| distribution:
    # it is so heavy-tailed that the top bin holds a couple of pixels, and equalizing bin
    # counts downward then throws away ~99.99% of the pool (691k pixels -> ~96). Two
    # corrections, both needed:
    #   (a) cap the top edge at `balance_cap` (default: the p99.9 of the scores) and lump
    #       everything above it into the last bin, so the degenerate extreme tail cannot
    #       drive the scheme;
    #   (b) place the remaining edges on quantiles rather than at equal widths, so bins
    #       hold comparable numbers of distinct pixels.
    bin_edges, cap = build_bz_balance_bin_edges(
        candidate_scores, n_bins=n_bins, balance_cap=balance_cap, bin_scale=bin_scale
    )

    bin_ids = np.digitize(candidate_scores, bin_edges[1:-1], right=False)
    bin_ids = np.clip(bin_ids, 0, bin_edges.size - 2)

    counts_before = {
        f"bin_{bin_idx}": int(np.sum(bin_ids == bin_idx))
        for bin_idx in range(bin_edges.size - 1)
    }
    occupied_bins = [count for count in counts_before.values() if count > 0]
    if not occupied_bins:
        raise ValueError("No occupied Bz bins found during balancing")

    # Equalize UPWARD (oversample rare bins with replacement) instead of downward. The
    # documented intent is that extreme-field pixels are "seen more often"; undersampling
    # the common bins to the rarest one achieves that ratio by discarding nearly the whole
    # dataset instead. Target defaults to the median occupied-bin count, which keeps the
    # selection the same order of magnitude as the pool.
    if oversample:
        target_per_bin = int(np.median(occupied_bins))
    else:
        target_per_bin = min(occupied_bins)
    target_per_bin = max(1, target_per_bin)

    selected_chunks = []
    counts_after = {}
    n_oversampled_bins = 0
    for bin_idx in range(bin_edges.size - 1):
        mask = bin_ids == bin_idx
        bin_indices = candidate_indices[mask]
        bin_count = int(bin_indices.size)
        if bin_count == 0:
            counts_after[f"bin_{bin_idx}"] = 0
            continue
        if bin_count > target_per_bin:
            chosen = rng.choice(bin_indices, size=target_per_bin, replace=False)
        elif oversample and bin_count < target_per_bin:
            # Draw with replacement: these are the rare strong-field pixels we want the
            # model to see more of. Replication adds no new information, so cap how far a
            # sparse bin may be inflated -- without this a bin holding a handful of pixels
            # gets replicated thousands of times and the model memorizes those pixels
            # instead of learning the regime.
            allowed = int(min(target_per_bin, bin_count * max_oversample_factor))
            chosen = rng.choice(bin_indices, size=max(allowed, bin_count), replace=True)
            n_oversampled_bins += 1
        else:
            chosen = bin_indices
        counts_after[f"bin_{bin_idx}"] = int(chosen.size)
        selected_chunks.append(chosen)

    if not selected_chunks:
        raise ValueError("Bz balancing produced no selected indices")

    selected = np.concatenate(selected_chunks)
    rng.shuffle(selected)
    counts_after["total_selected"] = int(selected.size)

    return selected.astype(np.int64), {
        "score_mode": score_mode,
        "tau_idx": None if tau_idx is None else int(tau_idx),
        "n_bins": int(bin_edges.size - 1),
        "score_min": score_min,
        "score_max": score_max,
        "bin_edges": [float(v) for v in bin_edges.tolist()],
        "counts_before": counts_before,
        "counts_after": counts_after,
        "total_candidates": int(candidate_indices.size),
        "target_per_bin": int(target_per_bin),
        "oversample": bool(oversample),
        "bin_scale": str(bin_scale),
        "balance_cap": float(cap),
        "n_oversampled_bins": int(n_oversampled_bins),
    }


# ============================================================================
# MuramStepDataset class - PyTorch Dataset for training
# ============================================================================
class MuramStepDataset(Dataset):
    """PyTorch Dataset of one MURaM step: normalized Stokes input and MHD targets per pixel.

    Every pixel becomes one sample. The constructor normalizes the whole step once with
    the fitted normalizers and keeps flat arrays; ``selected_flat_indices`` (from region
    or Bz balancing) then picks, and may repeat, the pixels to serve.

    Each item is a tuple of three tensors:

    * ``stokes_input``: float32, ``(2, n_wl)``, normalized I and V in that order;
    * ``mhd_targets``: float32, ``(3 * n_tau,)``, normalized T, then Vz, then Bz, each
      ordered by node as in the normalizer;
    * ``spatial_indices``: int64, ``(2,)``, the pixel's (x, y) in the step.

    Examples:
        >>> import contextlib, io, tempfile
        >>> rng = np.random.default_rng(0)
        >>> stokes = {"I": 1 + 0.1 * rng.random((3, 2, 5)), "V": 1e-3 * rng.random((3, 2, 5))}
        >>> mhd = {"T": 5000 + rng.random((3, 2, 4)), "Vz": rng.random((3, 2, 4)),
        ...        "Bz": 100 * rng.random((3, 2, 4))}
        >>> sn, mn = StokesNormalizer(), MhdNormalizer(n_tau=4, raw_values_dir=tempfile.mkdtemp())
        >>> sn.update(stokes); mn.update(mhd)
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     _ = sn.finalize(); _ = mn.finalize()
        >>> ds = MuramStepDataset(stokes, mhd, sn, mn, selected_flat_indices=np.array([5, 0, 5]))
        >>> x, y, xy = ds[0]
        >>> len(ds), tuple(x.shape), tuple(y.shape), xy.tolist()
        (3, (2, 5), (12,), [2, 1])
    """
    
    def __init__(
        self,
        stokes_data: Dict[str, np.ndarray],
        mhd_data: Dict[str, np.ndarray],
        stokes_normalizer: StokesNormalizer,
        mhd_normalizer: MhdNormalizer,
        selected_flat_indices: Optional[np.ndarray] = None,
        region_sampling_info: Optional[Dict[str, Any]] = None,
        bz_balance_info: Optional[Dict[str, Any]] = None,
    ):
        """Normalize one step and flatten it into per-pixel samples.

        Args:
            stokes_data: ``{"I", "V"}`` arrays of shape ``(nx, ny, n_wl)`` in continuum
                units. An optional ``"hinode_wl"`` entry is kept as ``self.hinode_wl``.
            mhd_data: ``{"T", "Vz", "Bz"}`` arrays of shape ``(nx, ny, n_tau)`` in K, km/s
                and G.
            stokes_normalizer: Fitted :class:`StokesNormalizer`.
            mhd_normalizer: Fitted :class:`MhdNormalizer`.
            selected_flat_indices: 1-D indices into the ``nx * ny`` flattened pixels; may
                repeat. ``None`` serves every pixel once.
            region_sampling_info: Stored as is, for logging.
            bz_balance_info: Stored as is, for logging.

        Raises:
            ValueError: If ``selected_flat_indices`` is not 1-D, is empty or is out of range.
        """
        self.nx, self.ny = stokes_data['I'].shape[:2]
        self.n_pixels_full = self.nx * self.ny
        self.region_sampling_info = region_sampling_info
        self.bz_balance_info = bz_balance_info
        self.hinode_wl = None
        if "hinode_wl" in stokes_data:
            self.hinode_wl = np.asarray(stokes_data["hinode_wl"], dtype=np.float32)
        
        # Normalize data
        norm_stokes = stokes_normalizer.transform(stokes_data)
        norm_mhd = mhd_normalizer.transform(mhd_data)
        
        # Flatten spatial dimensions: (nx, ny, ...) -> (nx*ny, ...)
        # Stokes input: (n_pixels, 2, nλ)
        I_flat = norm_stokes['I'].reshape(self.n_pixels_full, -1)  # (n_pixels, 112)
        V_flat = norm_stokes['V'].reshape(self.n_pixels_full, -1)
        self.stokes_input = np.stack([I_flat, V_flat], axis=1)  # (n_pixels, 2, 112)
        
        # MHD targets: concatenate T, Vz, Bz along feature dimension
        # Each has shape (n_pixels, 21) -> concatenate to (n_pixels, 63)
        T_flat = norm_mhd['T'].reshape(self.n_pixels_full, -1)
        Vz_flat = norm_mhd['Vz'].reshape(self.n_pixels_full, -1)
        Bz_flat = norm_mhd['Bz'].reshape(self.n_pixels_full, -1)
        self.mhd_targets = np.concatenate([T_flat, Vz_flat, Bz_flat], axis=1)
        
        # Store spatial indices for physics regularization
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing='ij')
        self.spatial_indices = np.stack([ix.ravel(), iy.ravel()], axis=1)  # (n_pixels_full, 2)

        if selected_flat_indices is not None:
            selected_flat_indices = np.asarray(selected_flat_indices, dtype=np.int64)
            if selected_flat_indices.ndim != 1:
                raise ValueError("selected_flat_indices must be 1D")
            if selected_flat_indices.size == 0:
                raise ValueError("selected_flat_indices is empty")
            if selected_flat_indices.min() < 0 or selected_flat_indices.max() >= self.n_pixels_full:
                raise ValueError(
                    "selected_flat_indices contains out-of-range values. "
                    f"Valid range is [0, {self.n_pixels_full - 1}]"
                )

            self.stokes_input = self.stokes_input[selected_flat_indices]
            self.mhd_targets = self.mhd_targets[selected_flat_indices]
            self.spatial_indices = self.spatial_indices[selected_flat_indices]

        self.n_pixels = int(self.stokes_input.shape[0])
        
    def __len__(self):
        """Number of samples served (selected pixels, counting repeats)."""
        return self.n_pixels
    
    def __getitem__(self, idx):
        """Return ``(stokes_input, mhd_targets, spatial_indices)`` tensors for sample ``idx``."""
        return (
            torch.from_numpy(self.stokes_input[idx]).float(),
            torch.from_numpy(self.mhd_targets[idx]).float(),
            torch.from_numpy(self.spatial_indices[idx]).long(),
        )


__all__ = [
    "MhdData",
    "StokesData",
    "MuramStepDataset",
    "build_granulation_polarization_masks",
    "build_balanced_region_indices",
    "build_bz_strength_balanced_indices",
]
