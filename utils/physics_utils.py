"""
Physics-based inversion and approximation methods for solar magnetometry.

This module provides classes for computing line-of-sight (LOS) magnetic field
and velocity from spectropolarimetric observations using weak-field approximation
(WFA) and Doppler shift methods.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import astropy.units as u
from astropy.constants import c, e, m_e

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from scipy.optimize import curve_fit


# NICOLE's `Continuum reference = 1` divides every profile by the HSRA disk-centre continuum
# at the middle of the synthesized range. The nicole_tau500 training cubes span
# 6300.5 + 300 * 0.01 A, so that normalization wavelength is 6302.0 A.
HSRA_REFERENCE_WAVELENGTH = 6302.0 * u.Angstrom

# Identifies the black-body algorithm in cache signatures; change it whenever the math changes.
TEMP_BB_METHOD = "planck_exact_hsra"

# Coefficients of NICOLE's `conhsra` polynomial fit to the HSRA disk-centre continuum, segment
# 6250 <= lambda < 8300 A (NICOLE/forward/forward.f90, array c4; max error 0.125 %). Lambda in
# Angstrom, result in erg s^-1 cm^-2 cm^-1 sr^-1, i.e. per unit wavelength (NICOLE manual,
# `Continuum reference`).
_CONHSRA_C4 = (4.861821e15, -2.2589885e12, 4.3764376e8, -39279.61444, 1.34388)
_CONHSRA_C4_RANGE_AA = (6250.0, 8300.0)


def hsra_continuum_intensity(wavelength: u.Quantity) -> u.Quantity:
    """HSRA disk-centre continuum intensity per unit wavelength, from NICOLE's `conhsra` fit."""
    lam = wavelength.to(u.Angstrom).value
    lo, hi = _CONHSRA_C4_RANGE_AA
    if not np.all((lam >= lo) & (lam < hi)):
        raise ValueError(f"conhsra port only covers {lo}-{hi} A, got {lam}")
    c0, c1, c2, c3, c4 = _CONHSRA_C4
    value = c0 + lam * (c1 + lam * (c2 + lam * (c3 + lam * c4)))
    return value * u.erg / (u.s * u.cm**2 * u.cm * u.sr)


def planck_brightness_temperature(intensity: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Exact inversion of the Planck function per unit wavelength: T such that B_lambda(T) = I."""
    from astropy.constants import h, k_B

    radiance = intensity * u.sr  # strip the per-steradian so the log argument is dimensionless
    argument = (2 * h * c**2 / (wavelength**5 * radiance)).to(u.dimensionless_unscaled).value
    return (h * c / (wavelength * k_B) / np.log1p(argument)).to(u.K)


def hsra_brightness_temperature(wavelength: u.Quantity = HSRA_REFERENCE_WAVELENGTH) -> u.Quantity:
    """Brightness temperature of the HSRA disk-centre continuum (~6125 K at 6302 A)."""
    return planck_brightness_temperature(hsra_continuum_intensity(wavelength), wavelength)


class ApproxInversions:
    """
    Class for computing approximate line-of-sight (LOS) magnetic field and velocity
    from Stokes polarimetric data using weak-field approximation (WFA) and Doppler methods.
    
    Attributes
    ----------
    stokes : dict
        Dictionary containing Stokes I, Q, U, V profiles with shape (nx, ny, n_wavelengths)
    wavelength : np.ndarray
        Wavelength array in Angstroms
    central_wavelength : astropy.units.Quantity
        Central wavelength of the spectral line (e.g., 6301.5*u.Angstrom)
    lande_factor : float
        Effective Landé g-factor for the spectral line
    blos : Optional[np.ndarray]
        Computed B_LOS map (2D)
    vlos : Optional[np.ndarray]
        Computed V_LOS map (2D)
    temperature : Optional[np.ndarray]
        Computed temperature map (2D)
    """
    
    def __init__(self, stokes: Dict[str, np.ndarray], wavelength: np.ndarray,
                 central_wavelength: u.Quantity, lande_factor: float):
        """
        Initialize ApproxInversions with Stokes parameters.
        
        Parameters
        ----------
        stokes : dict
            Dictionary with keys 'I', 'Q', 'U', 'V'. Each should be a 3D array
            with shape (nx, ny, n_wavelengths). Values can be Quantities or plain arrays.
        wavelength : np.ndarray or Quantity
            Wavelength axis. If Quantity, will be converted to Angstroms.
            If plain array, assumed to be in Angstroms.
        central_wavelength : Quantity
            Central wavelength of the spectral line (e.g., 6301.5*u.Angstrom)
        lande_factor : float
            Effective Landé g-factor (dimensionless)
        """
        # Store Stokes data (extract values if Quantities)
        self.stokes = {}
        for key, val in stokes.items():
            self.stokes[key] = val.value if hasattr(val, 'value') else val
        
        # Handle wavelength
        if hasattr(wavelength, 'unit'):
            self.wavelength = wavelength.to(u.Angstrom).value
        else:
            self.wavelength = np.asarray(wavelength)
        
        # Store parameters with units
        self.central_wavelength = central_wavelength
        self.lande_factor = lande_factor
        
        # Computed inversions (initialized as None)
        self.blos: Optional[np.ndarray] = None
        self.vlos: Optional[np.ndarray] = None
        self.temperature: Optional[np.ndarray] = None
    
    def compute_blos_wfa(self, wl_range: List[int]) -> u.Quantity:
        """
        Compute line-of-sight magnetic field using weak-field approximation (WFA).
        
        The WFA follows:
        V ≈ -(e / 4πm_e c) · λ₀² · g_λ · B_LOS · dI/dλ
        
        Parameters
        ----------
        wl_range : list of int
            [start_index, end_index] for the wavelength range to use in the WFA
            
        Returns
        -------
        astropy.units.Quantity
            Line-of-sight magnetic field map with shape (nx, ny) in Gauss
        """
        # WFA constant: e / (4π m_e c) with proper units
        wfa_constant = e.si / (4 * np.pi * m_e * c)
        wfa_constant = wfa_constant.to(1 / u.G / u.Angstrom)
        
        def estimate_B(dI_dl: np.ndarray, V: np.ndarray) -> u.Quantity:
            """
            Estimate B_LOS from intensity gradient and Stokes V using least-squares fit.
            
            Solves: V = (dI/dλ) · B_LOS + offset
            """
            ND = len(V)
            a = np.zeros([ND, 2])
            a[:, 0] = dI_dl
            a[:, 1] = 1.0
            b = V
            
            # Least-squares solution
            p = np.linalg.pinv(a) @ b
            
            # Compute B_LOS from the fit coefficient
            B = -p[0] * u.Angstrom / (wfa_constant * self.central_wavelength**2 * self.lande_factor)
            return B
        
        NX = self.stokes["I"].shape[0]
        NY = self.stokes["I"].shape[1]
        B = np.zeros([NX, NY])
        
        # Compute B_LOS for each pixel
        for i in range(NX):
            for j in range(NY):
                # Extract spectral range
                I_slice = self.stokes["I"][i, j, wl_range[0]:wl_range[1]]
                V_slice = self.stokes["V"][i, j, wl_range[0]:wl_range[1]]
                wl_slice = self.wavelength[wl_range[0]:wl_range[1]]
                
                # Compute intensity gradient
                dI_dl = np.gradient(I_slice) / np.gradient(wl_slice)
                
                # Estimate B at this pixel
                local_B = estimate_B(dI_dl=dI_dl, V=V_slice)
                B[i, j] = local_B.value
        
        self.blos = B * u.G
        return self.blos
    
    def compute_vlos_doppler(self, wl_range: List[int]) -> u.Quantity:
        """
        Compute line-of-sight velocity using Doppler shift method with Gaussian fitting.
        
        The velocity is computed by fitting a Gaussian to the Stokes I profile
        to find the line center wavelength, then applying the Doppler formula:
        v_LOS = (λ_fit - λ₀) / λ₀ · c

        Sign convention: positive v_LOS means a redshifted line core. This matches the
        sign of MURaM's Vz, which is what the network is trained against, and the two must
        agree for the Doppler physics term to reinforce the MSE rather than fight it.
        The opposite convention was in place until it was caught empirically: the
        approximation correlated -0.965 with MURaM's Vz at log(tau) = -1.0, and the
        doppler_only ablation arm drove the predicted velocity to correlation -0.951 at
        exactly that height (its vlos_target_logtau) while leaving every other height
        untouched.
        
        Parameters
        ----------
        wl_range : list of int
            [start_index, end_index] for the wavelength range to use for Gaussian fitting
            
        Returns
        -------
        astropy.units.Quantity
            Line-of-sight velocity map with shape (nx, ny) in km/s
        """
        def gaussian_func(x, a, x0, sigma, offset):
            """Gaussian function for fitting absorption line profile."""
            return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset
        
        NX = self.stokes["I"].shape[0]
        NY = self.stokes["I"].shape[1]
        vlos_map = np.zeros((NX, NY))
        
        # Extract wavelength range for fitting
        wl_fit = self.wavelength[:wl_range[1]]
        central_wl_value = self.central_wavelength.to(u.Angstrom).value
        
        # Compute velocity for each pixel
        for ix in range(NX):
            for iy in range(NY):
                I_profile = self.stokes["I"][ix, iy, :wl_range[1]]
                
                try:
                    # Fit Gaussian to Stokes I profile
                    popt, _ = curve_fit(
                        gaussian_func, 
                        wl_fit, 
                        I_profile,
                        p0=[1.0, central_wl_value, 0.1, 0.0],
                        maxfev=5000
                    )
                    x0_fit = popt[1]  # Fitted line center wavelength
                    
                    # Compute velocity from Doppler shift
                    # v_LOS = c * (λ_fit - λ₀) / λ₀   (positive = redshift, as in MURaM Vz)
                    delta_wl = x0_fit - central_wl_value
                    v_los = (c * delta_wl / central_wl_value).to(u.km / u.s)
                    vlos_map[ix, iy] = v_los.value
                    
                except (RuntimeError, ValueError):
                    # If fit fails, set to NaN or zero
                    vlos_map[ix, iy] = np.nan
        
        self.vlos = vlos_map * u.km / u.s
        return self.vlos
    
    def compute_temperature_blackbody(
        self,
        cont_indices: List[int] = [108, 109, 110, 111],
        continuum_wavelength: Optional[u.Quantity] = None,
        hsra_normalized: bool = True,
    ) -> u.Quantity:
        """
        Estimate temperature from the Stokes I continuum as a Planck brightness temperature.

        The profiles are assumed normalized to the HSRA disk-centre continuum at
        `HSRA_REFERENCE_WAVELENGTH` (NICOLE's `Continuum reference = 1`), so the absolute
        intensity is I_abs = I_norm * I_HSRA and the temperature is the exact inversion

            T = (h c / λ_c k_B) / ln(1 + 2 h c² / (λ_c⁵ I_abs)),

        with no linearization and no anchor temperature: I_norm = 1 returns the HSRA
        brightness temperature (~6125 K at 6302 Å).

        Parameters
        ----------
        cont_indices : list of int
            Wavelength indices averaged as the continuum. The default is the red end of the
            112-point Hinode grid (6303.19-6303.26 Å), the highest points of the quiet-Sun
            profile; the blue end still sits in the wing of Fe I 6301.5 Å.
        continuum_wavelength : astropy.units.Quantity, optional
            Wavelength at which the Planck function is inverted. If None, the mean wavelength
            of `cont_indices`.
        hsra_normalized : bool
            Must be True: the inversion is only meaningful for HSRA-normalized profiles.

        Returns
        -------
        astropy.units.Quantity
            Temperature map with shape (nx, ny) in Kelvin.

        Notes
        -----
        Assumes LTE (I_c ≈ B_λ(T)) and disk centre (μ ≈ 1). This is a brightness temperature:
        by Eddington-Barbier it matches T near τ ≈ 2/3 (log τ ≈ -0.2), not at log τ = 0.
        """
        if not hsra_normalized:
            raise ValueError(
                "compute_temperature_blackbody needs profiles normalized to the HSRA "
                "disk-centre continuum (NICOLE 'Continuum reference = 1')."
            )

        if continuum_wavelength is None:
            continuum_wavelength = np.mean(self.wavelength[cont_indices]) * u.Angstrom

        # Per-pixel continuum in HSRA units, shape (nx, ny)
        I_norm_c = np.mean(self.stokes["I"][:, :, cont_indices], axis=2)

        # NICOLE divides by a single reference value per region, so the same factor applies
        # at every wavelength of the window
        I_abs = I_norm_c * hsra_continuum_intensity(HSRA_REFERENCE_WAVELENGTH)

        self.temperature = planck_brightness_temperature(I_abs, continuum_wavelength)
        return self.temperature
    
    def compare_with_mhd(self, mhd_od_data: Dict[str, np.ndarray],
                         approximation: str = "blos",
                         logtau_values: Optional[np.ndarray] = None,
                         figsize_metrics: Tuple[int, int] = (14, 6),
                         figsize_maps: Tuple[int, int] = (18, 6),
                         save_dir: Optional[str] = None) -> Dict[str, float]:
        """
        Compare approximation (B_LOS or V_LOS) with MHD data at different optical depths.
        
        Computes and plots:
        - Relative Root Mean Square Error (RRMSE) vs optical depth
        - Pearson correlation vs optical depth
        - Spatial maps at heights of best RRMSE and correlation
        
        Parameters
        ----------
        mhd_od_data : dict
            Dictionary with MHD data remapped to optical depth coordinates.
            Should contain 'Bz' for blos or 'Vz' for vlos.
        approximation : str
            "blos" or "vlos" - which approximation to compare
        logtau_values : np.ndarray, optional
            Array of log(τ) values corresponding to the optical depth levels.
            If None, uses indices.
        figsize_metrics : tuple
            Figure size for RRMSE and correlation plots
        figsize_maps : tuple
            Figure size for spatial comparison maps
        save_dir : str or Path, optional
            Directory to save figures
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'min_rrmse': minimum RRMSE value
            - 'min_rrmse_logtau': optical depth of minimum RRMSE
            - 'max_corr': maximum correlation value
            - 'max_corr_logtau': optical depth of maximum correlation
        """
        # Select approximation
        if approximation.lower() == "blos":
            if self.blos is None:
                raise ValueError("B_LOS not computed. Call compute_blos_wfa() first.")
            approx_map = self.blos.value if hasattr(self.blos, 'value') else self.blos
            mhd_key = "Bz"
            quantity_name = r"$B_{\text{LOS}}$"
            unit_str = "G"
        elif approximation.lower() == "vlos":
            if self.vlos is None:
                raise ValueError("V_LOS not computed. Call compute_vlos_doppler() first.")
            approx_map = self.vlos.value if hasattr(self.vlos, 'value') else self.vlos
            mhd_key = "Vz"
            quantity_name = r"$v_{\text{LOS}}$"
            unit_str = "km/s"
        else:
            raise ValueError("approximation must be 'blos' or 'vlos'")
        
        # Extract MHD data
        if mhd_key not in mhd_od_data:
            raise ValueError(f"'{mhd_key}' not found in MHD data")
        
        mhd_data = mhd_od_data[mhd_key]
        mhd_data = mhd_data.value if hasattr(mhd_data, 'value') else mhd_data
        
        nz = mhd_data.shape[2]
        
        # Set default logtau if not provided
        if logtau_values is None:
            logtau_values = np.arange(nz)
        
        # Compute metrics at each height
        rrmse_values = []
        correlation_values = []
        
        for k in range(nz):
            mhd_at_k = mhd_data[:, :, k]
            
            # RRMSE
            rmse = np.sqrt(np.mean((approx_map - mhd_at_k)**2))
            rrmse = rmse / np.mean(np.abs(mhd_at_k))
            rrmse_values.append(rrmse)
            
            # Pearson correlation
            corr, _ = pearsonr(approx_map.flatten(), mhd_at_k.flatten())
            correlation_values.append(corr)
        
        # Find best heights
        min_rrmse_idx = np.argmin(rrmse_values)
        max_corr_idx = np.argmax(correlation_values)
        min_rrmse = rrmse_values[min_rrmse_idx]
        max_corr = correlation_values[max_corr_idx]
        min_logtau_rrmse = logtau_values[min_rrmse_idx]
        max_logtau_corr = logtau_values[max_corr_idx]
        
        # Plot metrics
        fig, ax = plt.subplots(1, 2, figsize=figsize_metrics)
        
        ax[0].plot(logtau_values, rrmse_values, marker='o', linewidth=2, markersize=8, color='C0')
        ax[0].axvline(min_logtau_rrmse, color='red', linestyle='--', alpha=0.7, label=f'Min RRMSE at τ={min_logtau_rrmse:.2f}')
        ax[0].set_xlabel(r'$\log \tau$', fontsize=12)
        ax[0].set_ylabel('RRMSE', fontsize=12)
        ax[0].set_title(f'Relative RMS Error vs Optical Depth ({quantity_name})', fontsize=12)
        ax[0].grid(True, alpha=0.3)
        ax[0].legend()
        
        ax[1].plot(logtau_values, correlation_values, marker='s', linewidth=2, markersize=8, color='C1')
        ax[1].axvline(max_logtau_corr, color='red', linestyle='--', alpha=0.7, label=f'Max Corr at τ={max_logtau_corr:.2f}')
        ax[1].set_xlabel(r'$\log \tau$', fontsize=12)
        ax[1].set_ylabel('Pearson Correlation', fontsize=12)
        ax[1].set_title(f'Correlation vs Optical Depth ({quantity_name})', fontsize=12)
        ax[1].grid(True, alpha=0.3)
        ax[1].legend()
        
        plt.tight_layout()
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            fname = f"{approximation.lower()}_metrics_vs_logtau.png"
            fig.savefig(save_path / fname, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path / fname}")
        plt.show()
        
        # Plot spatial maps
        fig, ax = plt.subplots(1, 3, figsize=figsize_maps)
        fig.suptitle(
            f'{quantity_name} Comparison: '
            f'Best RRMSE={min_rrmse:.4f} at τ={min_logtau_rrmse:.2f} | '
            f'Best Corr={max_corr:.4f} at τ={max_logtau_corr:.2f}',
            fontsize='x-large'
        )
        
        # Approximation map
        vmin = np.percentile(approx_map, 1)
        vmax = np.percentile(approx_map, 99)
        im_approx = ax[0].imshow(approx_map, origin='lower', cmap='RdBu_r', vmin=vmin, vmax=vmax)
        ax[0].set_title(f'Approximation {quantity_name} Map')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        divider_approx = make_axes_locatable(ax[0])
        cax_approx = divider_approx.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im_approx, cax=cax_approx, label=f'{unit_str}')
        
        # MHD at lowest RRMSE
        vmin_mhd = np.percentile(mhd_data[:, :, min_rrmse_idx], 1)
        vmax_mhd = np.percentile(mhd_data[:, :, min_rrmse_idx], 99)
        im0 = ax[1].imshow(mhd_data[:, :, min_rrmse_idx], origin='lower', cmap='RdBu_r', vmin=vmin_mhd, vmax=vmax_mhd)
        ax[1].set_title(
            f'MHD {mhd_key} at τ={min_logtau_rrmse:.2f}\n(RRMSE={min_rrmse:.4f})',
            fontsize=11
        )
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        divider0 = make_axes_locatable(ax[1])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im0, cax=cax0, label=f'{unit_str}')
        
        # MHD at highest correlation
        vmin_mhd_corr = np.percentile(mhd_data[:, :, max_corr_idx], 1)
        vmax_mhd_corr = np.percentile(mhd_data[:, :, max_corr_idx], 99)
        im1 = ax[2].imshow(mhd_data[:, :, max_corr_idx], origin='lower', cmap='RdBu_r', vmin=vmin_mhd_corr, vmax=vmax_mhd_corr)
        ax[2].set_title(
            f'MHD {mhd_key} at τ={max_logtau_corr:.2f}\n(Correlation={max_corr:.4f})',
            fontsize=11
        )
        ax[2].set_xticks([])
        ax[2].set_yticks([])
        divider1 = make_axes_locatable(ax[2])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(im1, cax=cax1, label=f'{unit_str}')
        
        plt.tight_layout()
        if save_dir:
            fname = f"{approximation.lower()}_spatial_comparison.png"
            fig.savefig(Path(save_dir) / fname, dpi=150, bbox_inches='tight')
            print(f"Saved: {Path(save_dir) / fname}")
        plt.show()
        
        # Return summary metrics
        return {
            'min_rrmse': min_rrmse,
            'min_rrmse_logtau': min_logtau_rrmse,
            'max_corr': max_corr,
            'max_corr_logtau': max_logtau_corr,
            'rrmse_values': np.array(rrmse_values),
            'correlation_values': np.array(correlation_values),
        }

# ============================================================================
# EXAMPLE USAGE (INversions through approximations)
# ============================================================================
"""
from physics_utils import ApproxInversions

# Initialize with Stokes data
inversions = ApproxInversions(
    stokes=stokes.data,  # dict with 'I', 'V', etc.
    wavelength=stokes.wl,
    central_wavelength=6301.5*u.Angstrom,
    lande_factor=1.67
)

# Compute B_LOS
blos = inversions.compute_blos_wfa(wl_range=[15, 60])

# Compute V_LOS
vlos = inversions.compute_vlos_doppler(wl_range=[15, 60])

# Compute temperature
temperature = inversions.compute_temperature_blackbody()

# Compare with MHD at different heights
metrics = inversions.compare_with_mhd(
    mhd_od_data=mhd.od_data,
    approximation="blos",
    logtau_values=new_logtau,
    save_dir="./results/"
)

print(f"Best RRMSE: {metrics['min_rrmse']:.4f} at τ={metrics['min_rrmse_logtau']:.2f}")
print(f"Best Corr: {metrics['max_corr']:.4f} at τ={metrics['max_corr_logtau']:.2f}")
"""

