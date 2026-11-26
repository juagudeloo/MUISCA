"""Fast physical approximations used as targets of the physics-informed loss terms.

Each approximation maps the Stokes profiles of a pixel to a single value of one
atmospheric quantity, which the training then compares with the network prediction at
one chosen optical depth:

* weak-field approximation (WFA): Stokes V vs dI/dlambda of Fe I 6301.5 A gives B_LOS (G);
* Doppler: the centre of a Gaussian fitted to Stokes I of Fe I 6301.5 A gives v_LOS (km/s);
* black body: exact Planck inversion of the continuum gives a brightness temperature (K).

The module-level helpers reproduce NICOLE's HSRA disk-centre continuum (``conhsra``), which
is the reference that the training Stokes profiles are normalized to.

Examples:
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from utils.hinode_wavelengths import hinode_wavelength_grid
    >>> wl = hinode_wavelength_grid()
    >>> I = 1 - 0.6 * np.exp(-(wl - 6301.52) ** 2 / (2 * 0.05 ** 2))   # line redshifted by 20 mA
    >>> inv = ApproxInversions({"I": I[None, None], "V": np.zeros((1, 1, wl.size))},
    ...                        wl, 6301.5 * u.Angstrom, lande_factor=1.67)
    >>> round(float(inv.compute_vlos_doppler([15, 60]).value[0, 0]), 3)   # km/s, + = redshift
    0.951
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
    """HSRA disk-centre continuum intensity per unit wavelength, from NICOLE's ``conhsra`` fit.

    A Python port of the 6250-8300 A segment of NICOLE's polynomial fit to the HSRA
    continuum (``NICOLE/forward/forward.f90``, array ``c4``; max error 0.125 %). It is
    the absolute value that NICOLE's ``Continuum reference = 1`` divides by.

    Args:
        wavelength: Quantity (scalar or array) with length units.

    Returns:
        Intensity in erg s^-1 cm^-2 cm^-1 sr^-1 (per unit wavelength).

    Raises:
        ValueError: If any wavelength is outside [6250, 8300) A.

    Examples:
        >>> import astropy.units as u
        >>> f"{hsra_continuum_intensity(6302.0 * u.Angstrom).value:.4e}"
        '2.9538e+14'
    """
    lam = wavelength.to(u.Angstrom).value
    lo, hi = _CONHSRA_C4_RANGE_AA
    if not np.all((lam >= lo) & (lam < hi)):
        raise ValueError(f"conhsra port only covers {lo}-{hi} A, got {lam}")
    c0, c1, c2, c3, c4 = _CONHSRA_C4
    value = c0 + lam * (c1 + lam * (c2 + lam * (c3 + lam * c4)))
    return value * u.erg / (u.s * u.cm**2 * u.cm * u.sr)


def planck_brightness_temperature(intensity: u.Quantity, wavelength: u.Quantity) -> u.Quantity:
    """Exact inverse of the Planck function per unit wavelength (no linearization).

    Solves ``B_lambda(T) = I`` for T::

        T = (h c / (lambda k_B)) / ln(1 + 2 h c^2 / (lambda^5 I)).

    Args:
        intensity: Specific intensity per unit wavelength, e.g. erg s^-1 cm^-2 cm^-1 sr^-1.
            Scalar or array Quantity.
        wavelength: Wavelength Quantity.

    Returns:
        Brightness temperature in K, with the shape of ``intensity``.

    Examples:
        >>> import astropy.units as u
        >>> from astropy.modeling.physical_models import BlackBody
        >>> lam = 6303.2 * u.Angstrom
        >>> b = BlackBody(temperature=6000 * u.K, scale=1 * u.erg / (u.s * u.cm**2 * u.cm * u.sr))
        >>> round(planck_brightness_temperature(b(lam), lam).to_value(u.K), 3)
        6000.0
    """
    from astropy.constants import h, k_B

    radiance = intensity * u.sr  # strip the per-steradian so the log argument is dimensionless
    argument = (2 * h * c**2 / (wavelength**5 * radiance)).to(u.dimensionless_unscaled).value
    return (h * c / (wavelength * k_B) / np.log1p(argument)).to(u.K)


def hsra_brightness_temperature(wavelength: u.Quantity = HSRA_REFERENCE_WAVELENGTH) -> u.Quantity:
    """Brightness temperature of the HSRA disk-centre continuum.

    It is the temperature that the black-body approximation assigns to a pixel with
    HSRA-normalized continuum exactly 1.

    Args:
        wavelength: Wavelength Quantity. Defaults to ``HSRA_REFERENCE_WAVELENGTH``
            (6302.0 A, the NICOLE normalization wavelength of the training cubes).

    Returns:
        Temperature in K.

    Examples:
        >>> round(hsra_brightness_temperature().to_value("K"))
        6125
    """
    return planck_brightness_temperature(hsra_continuum_intensity(wavelength), wavelength)


class ApproxInversions:
    """Per-pixel physical approximations of B_LOS, v_LOS and T from Stokes profiles.

    Every ``compute_*`` method returns a 2D map, shape ``(nx, ny)``, and stores it on the
    instance (``blos``, ``vlos``, ``temperature``). Training computes these maps once per
    simulation step and caches them (``approximations/*`` in the step HDF5 cache).

    Attributes:
        stokes: Dict of plain arrays ``(nx, ny, n_wl)``, continuum-normalized. Quantities
            are stripped to their values.
        wavelength: Wavelength axis in Angstrom, shape ``(n_wl,)``.
        central_wavelength: Rest wavelength of the line used by the WFA and Doppler terms
            (Fe I 6301.5 A in training).
        lande_factor: Effective Landé factor of that line (1.67 for Fe I 6301.5 A).
        blos: Last WFA map, Quantity in G, or None.
        vlos: Last Doppler map, Quantity in km/s, or None.
        temperature: Last black-body map, Quantity in K, or None.

    Examples:
        >>> import numpy as np
        >>> import astropy.units as u
        >>> wl = np.linspace(6300.9, 6303.3, 112)
        >>> inv = ApproxInversions({"I": np.ones((2, 3, 112)), "V": np.zeros((2, 3, 112))},
        ...                        wl, 6301.5 * u.Angstrom, lande_factor=1.67)
        >>> inv.compute_temperature_blackbody(cont_indices=[108, 109, 110, 111]).shape
        (2, 3)
    """

    def __init__(self, stokes: Dict[str, np.ndarray], wavelength: np.ndarray,
                 central_wavelength: u.Quantity, lande_factor: float):
        """Store the profiles and line parameters; no computation happens here.

        Args:
            stokes: Dict with at least ``'I'``, plus ``'V'`` for the WFA. Each entry is an
                array or Quantity of shape ``(nx, ny, n_wl)``.
            wavelength: Wavelength axis, either a Quantity (converted to Angstrom) or a
                plain array already in Angstrom.
            central_wavelength: Rest wavelength of the line, as a Quantity.
            lande_factor: Effective Landé factor (dimensionless).
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
        """B_LOS from the weak-field approximation, by per-pixel least squares.

        The WFA (e.g. del Toro Iniesta 2016) states::

            V(lambda) = -C lambda0^2 g_eff B_LOS dI/dlambda,   C = e / (4 pi m_e c^2)
                      = 4.67e-13 A^-1 G^-1.

        For each pixel, ``V = p0 * dI/dlambda + p1`` is fitted over the window (the
        offset ``p1`` absorbs any constant V bias), and then
        ``B_LOS = -p0 / (C lambda0^2 g_eff)``. ``dI/dlambda`` uses ``np.gradient`` on the
        window's own wavelengths.

        Args:
            wl_range: ``[start, stop)`` wavelength indices of the window. Training uses
                ``[15, 60]``, i.e. 6301.20-6302.14 A on the Hinode grid, which covers
                Fe I 6301.5 A.

        Returns:
            B_LOS map, shape ``(nx, ny)``, in G. Also stored in ``self.blos``.

        Examples:
            >>> import numpy as np
            >>> import astropy.units as u
            >>> from astropy.constants import c, e, m_e
            >>> from utils.hinode_wavelengths import hinode_wavelength_grid
            >>> wl = hinode_wavelength_grid()
            >>> I = 1 - 0.6 * np.exp(-(wl - 6301.5) ** 2 / (2 * 0.05 ** 2))
            >>> C = (e.si / (4 * np.pi * m_e * c)).to(1 / u.G / u.Angstrom).value
            >>> V = -C * 6301.5 ** 2 * 1.67 * 500.0 * np.gradient(I) / np.gradient(wl)
            >>> inv = ApproxInversions({"I": I[None, None], "V": V[None, None]}, wl,
            ...                        6301.5 * u.Angstrom, 1.67)
            >>> round(float(inv.compute_blos_wfa([15, 60]).value[0, 0]), 3)
            500.0
        """
        # WFA constant: e / (4π m_e c) with proper units
        wfa_constant = e.si / (4 * np.pi * m_e * c)
        wfa_constant = wfa_constant.to(1 / u.G / u.Angstrom)
        
        def estimate_B(dI_dl: np.ndarray, V: np.ndarray) -> u.Quantity:
            """Least-squares fit of ``V = p0 * dI/dlambda + p1``, returning ``B = -p0 / (C lambda0^2 g)`` in G.

            Args:
                dI_dl: dI/dlambda over the window, per Angstrom.
                V: Stokes V over the window.

            Returns:
                B_LOS of the pixel as a Quantity in G.
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
        """v_LOS from the centre of a Gaussian fitted to Stokes I (Doppler shift).

        Per pixel, ``a exp(-(x - x0)^2 / (2 sigma^2)) + offset`` is fitted with
        ``scipy.optimize.curve_fit``, starting from ``[1, lambda0, 0.1, 0]``, and then::

            v_LOS = c (x0 - lambda0) / lambda0.

        Sign convention: positive v_LOS means a redshifted line core (motion away from the
        observer), the same as the training target v_LOS = -Vz (MURaM's Vz is positive
        upward; NICOLE synthesizes the profiles with v_LOS = -Vz since Phase 1a). The two
        must agree for the Doppler physics term to reinforce the MSE instead of fighting it.

        History: an opposite sign of this approximation was caught empirically (it
        correlated -0.965 with the target at log(tau) = -1.0, and the doppler_only arm drove
        the predicted velocity to correlation -0.951 at that height). Profiles from before
        Phase 1a were synthesized with v_LOS = +Vz, so on them the target is +Vz and the
        same redshift-positive approximation still agrees with it.

        Args:
            wl_range: Only ``wl_range[1]`` is used. The fit runs over indices
                ``[0, wl_range[1])``, i.e. 6300.87-6302.14 A for the training value
                ``[15, 60]``, which also contains the blue wing of Fe I 6302.5 A.

        Returns:
            v_LOS map, shape ``(nx, ny)``, in km/s. NaN where the fit fails. Also stored in
            ``self.vlos``.

        Examples:
            >>> import numpy as np
            >>> import astropy.units as u
            >>> from utils.hinode_wavelengths import hinode_wavelength_grid
            >>> wl = hinode_wavelength_grid()
            >>> I = 1 - 0.6 * np.exp(-(wl - 6301.48) ** 2 / (2 * 0.05 ** 2))   # 20 mA blueshift
            >>> inv = ApproxInversions({"I": I[None, None], "V": np.zeros((1, 1, wl.size))}, wl,
            ...                        6301.5 * u.Angstrom, 1.67)
            >>> round(float(inv.compute_vlos_doppler([15, 60]).value[0, 0]), 3)
            -0.951
        """
        def gaussian_func(x, a, x0, sigma, offset):
            """Gaussian plus offset; ``a`` goes negative when fitted to an absorption line.

            Args:
                x: Wavelengths (A).
                a: Amplitude.
                x0: Centre (A).
                sigma: Width (A).
                offset: Constant level.

            Returns:
                Model values at ``x``.
            """
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
        """Brightness temperature of the Stokes I continuum (exact Planck inversion).

        The profiles are assumed normalized to the HSRA disk-centre continuum at
        ``HSRA_REFERENCE_WAVELENGTH`` (NICOLE's ``Continuum reference = 1``), so the
        absolute intensity is ``I_abs = I_norm * I_HSRA`` and::

            T = (h c / (lambda_c k_B)) / ln(1 + 2 h c^2 / (lambda_c^5 I_abs)).

        There is no linearization and no anchor temperature: ``I_norm = 1`` gives the HSRA
        brightness temperature (~6125 K at 6302 A).

        Args:
            cont_indices: Wavelength indices averaged as the continuum. The default is the
                red end of the 112-point Hinode grid (6303.19-6303.26 A), the highest points
                of the quiet-Sun profile. The blue end still sits in the wing of
                Fe I 6301.5 A.
            continuum_wavelength: Wavelength where the Planck function is inverted.
                ``None`` uses the mean wavelength of ``cont_indices``.
            hsra_normalized: Must be True. The inversion is only meaningful for
                HSRA-normalized profiles.

        Returns:
            Temperature map, shape ``(nx, ny)``, in K. Also stored in ``self.temperature``.

        Raises:
            ValueError: If ``hsra_normalized`` is False.

        Note:
            Assumes LTE (``I_c = B_lambda(T)``) and disk centre (mu ~ 1). Being a brightness
            temperature, by Eddington-Barbier it matches T near tau ~ 2/3
            (log tau ~ -0.2), not at log tau = 0, hence the comparison height of the
            Black Body flavor. In cold umbrae the red-end points sit next to a molecular
            blend at 6303.1 A (Castellanos Durán et al. 2024, Fig. 11).

        Examples:
            >>> import numpy as np
            >>> import astropy.units as u
            >>> from utils.hinode_wavelengths import hinode_wavelength_grid
            >>> wl = hinode_wavelength_grid()
            >>> inv = ApproxInversions({"I": np.ones((1, 1, wl.size))}, wl, 6301.5 * u.Angstrom, 1.67)
            >>> round(float(inv.compute_temperature_blackbody().to_value(u.K)[0, 0]))   # at 6303.2 A
            6126
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
        """Compare the WFA or Doppler map with the simulation at every optical depth.

        For each node ``k`` it computes::

            RRMSE_k = sqrt(mean((approx - truth_k)^2)) / mean(|truth_k|)

        and the Pearson correlation, then plots both against log tau and shows the maps
        at the best-RRMSE and best-correlation heights. This is how the comparison
        height of each physics term was chosen.

        Args:
            mhd_od_data: Simulation remapped to optical depth, with ``'Bz'`` (G) or
                ``'Vz'`` (km/s) of shape ``(nx, ny, n_tau)``.
            approximation: ``'blos'`` (needs ``compute_blos_wfa`` first) or ``'vlos'``
                (needs ``compute_vlos_doppler`` first).
            logtau_values: log tau of each node. ``None`` uses the node indices.
            figsize_metrics: Figure size of the metrics-vs-log-tau plot, in inches.
            figsize_maps: Figure size of the maps figure, in inches.
            save_dir: If given, both figures are saved there as PNG.

        Returns:
            Dict with ``min_rrmse``, ``min_rrmse_logtau``, ``max_corr``,
            ``max_corr_logtau`` and the full per-node arrays ``rrmse_values`` and
            ``correlation_values``.

        Raises:
            ValueError: If the approximation is unknown or not computed yet, or the
                matching key is missing from ``mhd_od_data``.

        Examples:
            >>> metrics = inv.compare_with_mhd(mhd_od_data={"Bz": bz_tau}, approximation="blos",
            ...                                logtau_values=np.arange(-3.0, 1.41, 0.1))  # doctest: +SKIP
            >>> metrics["min_rrmse_logtau"]  # doctest: +SKIP
            -0.8
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

