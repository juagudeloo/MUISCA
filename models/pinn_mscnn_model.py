"""Physics-informed MSCNN: the MSCNN plus loss terms that compare its output with simple approximations.

Each physics term takes the network's prediction at one node (``single_height``) or
averaged over linear tau (``tau_averaged``), in physical units, and compares it with an
approximation computed once per step from the Stokes profiles of the same pixel
(:class:`utils.physics_utils.ApproxInversions`): the WFA for B_LOS (G), the Doppler shift
for v_LOS (km/s) and the black-body temperature (K). Each comparison is an RRMSE,
dimensionless, so the terms can be weighted against the MSE with plain numbers.

The total loss is ``MSE + lambda_wfa L_WFA + lambda_doppler L_Doppler + lambda_temp L_BB``.

Examples:
    Typical use inside the training loop::

        model.set_physics_context(mhd_norm, logtau, blos_approx=..., vlos_approx=..., temp_approx=...)
        out = model.compute_loss(model(stokes), targets, spatial_indices)
        out["loss"].backward()
"""

from typing import Dict, Optional, Tuple, Any, Literal
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import astropy.units as u

from models.mscnn_model import MSCNNInversionModel

# Overflow guard on the argument of sinh() when denormalizing Bz. float32 sinh overflows
# around 89; 20 already corresponds to ~|B| = 1e8 G with a typical per-tau B0, which is far
# beyond anything physical, so this only ever catches a diverging prediction and never
# interferes with gradients in the range the data occupies.
_SINH_ARG_LIMIT = 20.0


class PhysicsInformedMSCNN(MSCNNInversionModel):
    """MSCNN with physics regularization computed in physical units.

    Accepts every argument of :class:`models.mscnn_model.MSCNNInversionModel` plus the
    physics settings below.

    Note:
        * ``spatial_indices`` are the pixel's ``(x, y)`` as produced by
          :class:`utils.muram_data.MuramStepDataset`, and the approximation maps are
          ``(nx, ny)``, so ``approx[x, y]`` is the pixel's own value.
        * The Vz denormalization adds the node mean, but the normalizer's "centered"
          transform never subtracted it, so the Vz used by the Doppler term is offset by
          +mean(tau): at most 0.03 km/s on the paper nodes (strict xfail in
          ``tests/test_model.py``; fixed in reframe Phase 1).
        * ``tau_averaged`` weights the nodes by d(tau) in linear tau, so with nodes up to
          log tau = 1.4 the average is dominated by the deepest few nodes.
        * The Bz tail parameters (``B0_weight_start``, ``B1_weight_saturation``,
          ``huber_delta``) are loaded from the normalizer but no loss term uses them.
    """

    def __init__(
        self,
        *,
        lambda_wfa: float = 0.01,
        lambda_doppler: float = 0.01,
        lambda_temp: float = 0.01,
        blos_physics_mode: Literal['tau_averaged', 'single_height'] = 'tau_averaged',
        blos_target_logtau: Optional[float] = None,
        vlos_physics_mode: Literal['tau_averaged', 'single_height'] = 'tau_averaged',
        vlos_target_logtau: Optional[float] = None,
        temp_physics_mode: Literal['tau_averaged', 'single_height'] = 'single_height',
        temp_target_logtau: Optional[float] = 0.0,
        **kwargs: Any,
    ) -> None:
        """Build the network and store the physics settings.

        Args:
            lambda_wfa: Weight of the WFA B_LOS term (0 disables it).
            lambda_doppler: Weight of the Doppler v_LOS term (0 disables it).
            lambda_temp: Weight of the black-body temperature term (0 disables it).
            blos_physics_mode: ``"single_height"`` (prediction at ``blos_target_logtau``)
                or ``"tau_averaged"``.
            blos_target_logtau: Node for the WFA comparison (the nearest node is used).
                ``None`` uses the middle node.
            vlos_physics_mode: Same choice for the Doppler term.
            vlos_target_logtau: Node for the Doppler comparison; ``None`` uses the middle node.
            temp_physics_mode: Same choice for the black-body term.
            temp_target_logtau: Node for the black-body comparison; ``None`` uses
                log tau = 0.
            **kwargs: Passed to :class:`MSCNNInversionModel` (``scales``, ``input_length``...).

        Raises:
            ValueError: For an unknown physics mode.
        """
        super().__init__(**kwargs)
        
        # Validate physics_mode for each quantity
        valid_physics_modes = ['tau_averaged', 'single_height']
        if blos_physics_mode not in valid_physics_modes:
            raise ValueError(f"blos_physics_mode must be one of {valid_physics_modes}, got {blos_physics_mode}")
        if vlos_physics_mode not in valid_physics_modes:
            raise ValueError(f"vlos_physics_mode must be one of {valid_physics_modes}, got {vlos_physics_mode}")
        if temp_physics_mode not in valid_physics_modes:
            raise ValueError(f"temp_physics_mode must be one of {valid_physics_modes}, got {temp_physics_mode}")
        
        self.lambda_wfa = float(lambda_wfa)
        self.lambda_doppler = float(lambda_doppler)
        self.lambda_temp = float(lambda_temp)
        self.blos_physics_mode = blos_physics_mode
        self.blos_target_logtau = blos_target_logtau
        self.vlos_physics_mode = vlos_physics_mode
        self.vlos_target_logtau = vlos_target_logtau
        self.temp_physics_mode = temp_physics_mode
        self.temp_target_logtau = temp_target_logtau
        
        # Tail-loss parameters (loaded from normalizer)
        self.B0_weight_start = None  # P90 of |Bz|
        self.B1_weight_saturation = None  # P99.5 of |Bz|
        self.huber_delta = None  # Huber threshold
        
        # Physics computation state (set via set_physics_context)
        self.mhd_normalizer = None
        self.logtau_values = None
        self.blos_approx = None
        self.vlos_approx = None
        self.temp_approx = None
        self._tau_linear = None
        self._dtau = None
        self._integral_dtau = None
        self._blos_target_logtau_idx = None
        self._vlos_target_logtau_idx = None
        self._temp_target_logtau_idx = None

    def set_physics_context(
        self,
        mhd_normalizer: 'MhdNormalizer',
        logtau_values: np.ndarray,
        blos_approx: Optional[np.ndarray] = None,
        vlos_approx: Optional[np.ndarray] = None,
        temp_approx: Optional[np.ndarray] = None,
        blos_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        blos_target_logtau: Optional[float] = None,
        vlos_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        vlos_target_logtau: Optional[float] = None,
        temp_physics_mode: Optional[Literal['tau_averaged', 'single_height']] = None,
        temp_target_logtau: Optional[float] = None,
    ):
        """Give the model what the physics terms need for the current step.

        Stores the normalizer (to denormalize predictions), the log tau grid, the step's
        approximation maps (moved to the model's device), and resolves each single-height
        target to the nearest node of the grid. Optional mode and target arguments override
        the ones given to the constructor.

        Args:
            mhd_normalizer: Fitted :class:`utils.normalizer.MhdNormalizer` of the targets.
            logtau_values: The model's log tau nodes, increasing.
            blos_approx: WFA B_LOS map ``(nx, ny)`` in G.
            vlos_approx: Doppler v_LOS map ``(nx, ny)`` in km/s (NaN where the fit failed).
            temp_approx: Black-body temperature map ``(nx, ny)`` in K.
            blos_physics_mode: Override of the B_LOS mode.
            blos_target_logtau: Override of the B_LOS node.
            vlos_physics_mode: Override of the v_LOS mode.
            vlos_target_logtau: Override of the v_LOS node.
            temp_physics_mode: Override of the temperature mode.
            temp_target_logtau: Override of the temperature node.

        Raises:
            ValueError: For an unknown physics mode.

        Examples:
            >>> import contextlib, io, tempfile
            >>> import numpy as np, torch
            >>> from utils.normalizer import MhdNormalizer
            >>> rng = np.random.default_rng(0)
            >>> atm = {"T": 5000 + 100 * rng.standard_normal((6, 6, 3)), "Vz": rng.standard_normal((6, 6, 3)),
            ...        "Bz": 100 * rng.standard_normal((6, 6, 3))}
            >>> norm = MhdNormalizer(n_tau=3, raw_values_dir=tempfile.mkdtemp()); norm.update(atm)
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = norm.finalize()
            >>> model = PhysicsInformedMSCNN(scales=[1], n_tau_levels=3, lambda_wfa=1.0, lambda_doppler=0.0,
            ...                              lambda_temp=0.0, blos_physics_mode="single_height",
            ...                              blos_target_logtau=-1.0)
            >>> model.set_physics_context(norm, np.array([-2.0, -1.0, 0.0]), blos_approx=np.full((6, 6), 50.0))
            >>> float(model.blos_target_logtau), model._blos_target_logtau_idx
            (-1.0, 1)
        """
        self.mhd_normalizer = mhd_normalizer
        self.logtau_values = logtau_values
        
        # Update physics modes if provided
        valid_physics_modes = ['tau_averaged', 'single_height']
        if blos_physics_mode is not None:
            if blos_physics_mode not in valid_physics_modes:
                raise ValueError(f"blos_physics_mode must be one of {valid_physics_modes}")
            self.blos_physics_mode = blos_physics_mode
        if vlos_physics_mode is not None:
            if vlos_physics_mode not in valid_physics_modes:
                raise ValueError(f"vlos_physics_mode must be one of {valid_physics_modes}")
            self.vlos_physics_mode = vlos_physics_mode
        if temp_physics_mode is not None:
            if temp_physics_mode not in valid_physics_modes:
                raise ValueError(f"temp_physics_mode must be one of {valid_physics_modes}")
            self.temp_physics_mode = temp_physics_mode
        
        # Update target logtau values if provided
        if blos_target_logtau is not None:
            self.blos_target_logtau = blos_target_logtau
        if vlos_target_logtau is not None:
            self.vlos_target_logtau = vlos_target_logtau
        if temp_target_logtau is not None:
            self.temp_target_logtau = temp_target_logtau
        
        # Pre-compute tau integration quantities
        self._tau_linear = 10 ** logtau_values
        self._dtau = np.diff(self._tau_linear)
        self._integral_dtau = self._tau_linear[-1] - self._tau_linear[0]
        
        # Compute target logtau indices for single_height mode
        if self.blos_physics_mode == 'single_height':
            if self.blos_target_logtau is None:
                self._blos_target_logtau_idx = len(logtau_values) // 2
                self.blos_target_logtau = logtau_values[self._blos_target_logtau_idx]
            else:
                self._blos_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.blos_target_logtau)))
                self.blos_target_logtau = logtau_values[self._blos_target_logtau_idx]
        
        if self.vlos_physics_mode == 'single_height':
            if self.vlos_target_logtau is None:
                self._vlos_target_logtau_idx = len(logtau_values) // 2
                self.vlos_target_logtau = logtau_values[self._vlos_target_logtau_idx]
            else:
                self._vlos_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.vlos_target_logtau)))
                self.vlos_target_logtau = logtau_values[self._vlos_target_logtau_idx]
        
        if self.temp_physics_mode == 'single_height':
            if self.temp_target_logtau is None:
                # Default to log(tau) = 0.0 (photosphere)
                self._temp_target_logtau_idx = int(np.argmin(np.abs(logtau_values - 0.0)))
                self.temp_target_logtau = logtau_values[self._temp_target_logtau_idx]
            else:
                self._temp_target_logtau_idx = int(np.argmin(np.abs(logtau_values - self.temp_target_logtau)))
                self.temp_target_logtau = logtau_values[self._temp_target_logtau_idx]
        
        # Load tail-loss parameters from normalizer (new names + legacy fallback)
        B0_weight_start = getattr(mhd_normalizer, 'B0_weight_start', None)
        B1_weight_saturation = getattr(mhd_normalizer, 'B1_weight_saturation', None)
        huber_delta = getattr(mhd_normalizer, 'huber_delta', None)

        if B0_weight_start is None:
            B0_weight_start = getattr(mhd_normalizer, 'bz_B0_w', None)
            B1_weight_saturation = getattr(mhd_normalizer, 'bz_B1_w', None)
            huber_delta = getattr(mhd_normalizer, 'bz_delta', None)

        if B0_weight_start is not None and B1_weight_saturation is not None and huber_delta is not None:
            self.B0_weight_start = float(B0_weight_start)
            self.B1_weight_saturation = float(B1_weight_saturation)
            self.huber_delta = float(huber_delta)
        
        # Move approximations to device
        device = self._get_device()
        if blos_approx is not None:
            self.blos_approx = torch.tensor(blos_approx, dtype=torch.float32, device=device)
        if vlos_approx is not None:
            self.vlos_approx = torch.tensor(vlos_approx, dtype=torch.float32, device=device)
        if temp_approx is not None:
            self.temp_approx = torch.tensor(temp_approx, dtype=torch.float32, device=device)

    def _get_device(self) -> torch.device:
        """Device of the model's parameters (CPU if it has none)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _denormalize_predictions(self, predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convert normalized predictions to physical units with torch ops, so gradients flow.

        Per node, T and Vz are ``x std + mean`` (K, km/s) and Bz is
        ``B0 sinh(x std + mean)`` (G), mirroring :meth:`MhdNormalizer.denormalize`. The sinh
        argument is clamped to +-20 as an overflow guard only; unlike the numpy path no
        physical clip in G is applied, so runaway pixels keep their gradient.

        This used to round-trip through numpy, which cut the autograd graph: every physics
        term contributed zero gradient while still appearing in the logged totals.

        Args:
            predictions: ``(batch, 3 n_tau)`` in the order T, Vz, Bz, or ``(batch, n_tau, 3)``.

        Returns:
            ``{"T", "Vz", "Bz"}``, each ``(batch, n_tau)``.

        Raises:
            RuntimeError: If :meth:`set_physics_context` has not been called.
            ValueError: For a shape that is not one of the two above, or a node count that
                differs from the normalizer's.

        Examples:
            >>> import contextlib, io, tempfile
            >>> import numpy as np, torch
            >>> from utils.normalizer import MhdNormalizer
            >>> rng = np.random.default_rng(0)
            >>> atm = {"T": 5000 + 100 * rng.standard_normal((6, 6, 3)), "Vz": rng.standard_normal((6, 6, 3)),
            ...        "Bz": 100 * rng.standard_normal((6, 6, 3))}
            >>> norm = MhdNormalizer(n_tau=3, raw_values_dir=tempfile.mkdtemp()); norm.update(atm)
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = norm.finalize()
            >>> model = PhysicsInformedMSCNN(scales=[1], n_tau_levels=3, lambda_wfa=1.0, lambda_doppler=0.0,
            ...                              lambda_temp=0.0, blos_physics_mode="single_height",
            ...                              blos_target_logtau=-1.0)
            >>> model.set_physics_context(norm, np.array([-2.0, -1.0, 0.0]), blos_approx=np.full((6, 6), 50.0))
            >>> y = torch.tensor(norm.transform(atm)["T"].reshape(-1, 3)[:2], dtype=torch.float32)
            >>> pred = torch.cat([y, torch.zeros(2, 6)], dim=1)
            >>> out = model._denormalize_predictions(pred)
            >>> bool(torch.allclose(out["T"], torch.tensor(atm["T"].reshape(-1, 3)[:2], dtype=torch.float32)))
            True
        """
        if self.mhd_normalizer is None:
            raise RuntimeError("MHD normalizer not set. Call set_physics_context() first.")

        # Denormalize with torch ops so autograd can reach the network.
        #
        # This used to round-trip through numpy (predictions.detach().cpu().numpy(), then
        # MhdNormalizer.denormalize, then torch.tensor(...)). torch.tensor() on a numpy array
        # builds a fresh leaf, so the graph was severed: every physics loss computed from
        # these values had requires_grad=False and contributed exactly zero gradient. The
        # terms still appeared in the reported totals, which is why a run could show a large
        # WFA loss sitting flat for hundreds of epochs while only the MSE improved.
        #
        # Per-tau transforms mirror MhdNormalizer.denormalize:
        #   T, Vz : x * std + mean
        #   Bz    : B0 * sinh(x * std + mean)      (per-tau asinh scale)
        if predictions.ndim == 2:
            if predictions.shape[1] % 3 != 0:
                raise ValueError(
                    f"Expected predictions.shape[1] to be divisible by 3, got {predictions.shape[1]}"
                )
            n_tau = int(predictions.shape[1] // 3)
            t_norm = predictions[:, :n_tau]
            vz_norm = predictions[:, n_tau:2 * n_tau]
            bz_norm = predictions[:, 2 * n_tau:3 * n_tau]
        elif predictions.ndim == 3:
            if predictions.shape[2] != 3:
                raise ValueError(
                    f"Expected predictions.shape[2] == 3 for (batch, n_tau, 3), got {predictions.shape[2]}"
                )
            n_tau = int(predictions.shape[1])
            t_norm = predictions[:, :, 0]
            vz_norm = predictions[:, :, 1]
            bz_norm = predictions[:, :, 2]
        else:
            raise ValueError(
                f"Unsupported predictions ndim={predictions.ndim}; expected 2D or 3D tensor"
            )

        device = self._get_device()
        mean, std, b0 = self._denorm_params(n_tau, device, predictions.dtype)

        denorm_torch = {
            'T': t_norm * std['T'] + mean['T'],
            'Vz': vz_norm * std['Vz'] + mean['Vz'],
            # Bound the sinh argument before exponentiating. This is an overflow guard, not
            # the physical clip that MhdNormalizer.denormalize applies for reporting: the
            # bound sits far outside the range the training targets occupy, so gradients stay
            # alive everywhere a prediction could plausibly land. Clipping in Gauss instead
            # (as the numpy path does) would zero the gradient for exactly the runaway pixels
            # the physics terms exist to pull back.
            'Bz': b0 * torch.sinh(
                torch.clamp(bz_norm * std['Bz'] + mean['Bz'], min=-_SINH_ARG_LIMIT, max=_SINH_ARG_LIMIT)
            ),
        }

        return denorm_torch

    def _denorm_params(self, n_tau: int, device, dtype):
        """Per-node mean, std and B0 as tensors, cached per (n_tau, device, dtype)."""
        key = (n_tau, str(device), str(dtype))
        cached = getattr(self, "_denorm_param_cache", {}).get(key)
        if cached is not None:
            return cached

        stats = self.mhd_normalizer.final_stats
        if len(stats['Bz']) != n_tau:
            raise ValueError(
                f"Normalizer has {len(stats['Bz'])} tau levels but predictions carry {n_tau}. "
                "The checkpoint and the normalization statistics describe different grids."
            )

        def col(param, field):
            """One statistic of one quantity for every node, as a tensor."""
            return torch.tensor(
                [float(stats[param][i][field]) for i in range(n_tau)], dtype=dtype, device=device
            )

        mean = {p: col(p, 'mean') for p in ('T', 'Vz', 'Bz')}
        std = {p: col(p, 'std') for p in ('T', 'Vz', 'Bz')}
        b0 = torch.tensor(
            [float(self.mhd_normalizer.B0_transform_per_tau[i]) for i in range(n_tau)],
            dtype=dtype, device=device,
        )

        if not hasattr(self, "_denorm_param_cache"):
            self._denorm_param_cache = {}
        self._denorm_param_cache[key] = (mean, std, b0)
        return mean, std, b0

    def _compute_tau_averaged_blos(self, bz: torch.Tensor) -> torch.Tensor:
        """Average B_LOS over linear tau: trapezoid of Bz d(tau) divided by (tau_max - tau_min), in G."""
        device = self._get_device()
        
        # Trapezoidal integration
        bz_avg = (bz[:, :-1] + bz[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_bz = torch.sum(bz_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_bz / self._integral_dtau

    def _compute_tau_averaged_vlos(self, vz: torch.Tensor) -> torch.Tensor:
        """Average v_LOS over linear tau: trapezoid of Vz d(tau) divided by (tau_max - tau_min), in km/s."""
        device = self._get_device()
        
        # Trapezoidal integration
        vz_avg = (vz[:, :-1] + vz[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_vz = torch.sum(vz_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_vz / self._integral_dtau

    def _compute_tau_averaged_temperature(self, temp: torch.Tensor) -> torch.Tensor:
        """Average T over linear tau: trapezoid of T d(tau) divided by (tau_max - tau_min), in K."""
        device = self._get_device()
        
        # Trapezoidal integration
        temp_avg = (temp[:, :-1] + temp[:, 1:]) / 2  # (batch_size, 20)
        dtau_tensor = torch.tensor(self._dtau, dtype=torch.float32, device=device)
        integral_temp = torch.sum(temp_avg * dtau_tensor[None, :], dim=1)  # (batch_size,)
        
        return integral_temp / self._integral_dtau

    def _extract_at_logtau(self, values: torch.Tensor, target_idx: int) -> torch.Tensor:
        """Column ``target_idx`` of a ``(batch, n_tau)`` tensor."""
        return values[:, target_idx]

    def _compute_predicted_blos(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted B_LOS per sample (G), single-height or tau-averaged as configured."""
        if self.blos_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_blos(denorm_pred['Bz'])
        else:  # single_height
            if self._blos_target_logtau_idx is None:
                raise RuntimeError("B_LOS target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['Bz'], self._blos_target_logtau_idx)

    def _compute_predicted_vlos(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted v_LOS per sample (km/s), single-height or tau-averaged as configured."""
        if self.vlos_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_vlos(denorm_pred['Vz'])
        else:  # single_height
            if self._vlos_target_logtau_idx is None:
                raise RuntimeError("V_LOS target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['Vz'], self._vlos_target_logtau_idx)

    def _compute_predicted_temperature(self, denorm_pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Predicted T per sample (K), single-height or tau-averaged as configured."""
        if self.temp_physics_mode == 'tau_averaged':
            return self._compute_tau_averaged_temperature(denorm_pred['T'])
        else:  # single_height
            if self._temp_target_logtau_idx is None:
                raise RuntimeError("Temperature target logtau index not set. Call set_physics_context() first.")
            return self._extract_at_logtau(denorm_pred['T'], self._temp_target_logtau_idx)

    def _compute_rrmse(self, predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """RRMSE = sqrt(mean((pred - target)^2)) / mean(|target|), dimensionless.

        Examples:
            >>> import torch
            >>> m = PhysicsInformedMSCNN(scales=[1], n_tau_levels=1)
            >>> round(float(m._compute_rrmse(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 2.0]))), 6)
            0.5
        """
        mse = torch.mean((predictions - targets) ** 2)
        rmse = torch.sqrt(mse)
        mean_abs_target = torch.mean(torch.abs(targets))
        rrmse = rmse / (mean_abs_target + epsilon)
        return rrmse

    def _compute_wfa_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """RRMSE between the predicted B_LOS and the WFA map at each sample's pixel ``(x, y)``."""
        if self.blos_approx is None:
            raise RuntimeError("B_LOS approximation not set. Call set_physics_context() first.")
        
        # Compute B_LOS from predictions based on physics_mode
        pred_blos = self._compute_predicted_blos(denorm_pred)
        
        # Get corresponding WFA approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_blos = self.blos_approx[y_idx, x_idx]  # (batch_size,)
        
        # RRMSE loss in physical units
        return self._compute_rrmse(pred_blos, approx_blos)

    def _compute_doppler_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """RRMSE between the predicted v_LOS and the Doppler map at each sample's pixel, skipping NaN approximations.

        Returns 0 if every approximation in the batch is NaN.
        """
        if self.vlos_approx is None:
            raise RuntimeError("V_LOS approximation not set. Call set_physics_context() first.")
        
        # Compute V_LOS from predictions based on physics_mode
        pred_vlos = self._compute_predicted_vlos(denorm_pred)
        
        # Get corresponding Doppler approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_vlos = self.vlos_approx[y_idx, x_idx]  # (batch_size,)
        
        # Handle NaN values in approximations (from failed Gaussian fits)
        valid_mask = ~torch.isnan(approx_vlos)
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=self._get_device())
        
        # RRMSE loss in physical units (only valid pixels)
        return self._compute_rrmse(pred_vlos[valid_mask], approx_vlos[valid_mask])

    def _compute_temperature_loss(
        self,
        denorm_pred: Dict[str, torch.Tensor],
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """RRMSE between the predicted T and the black-body map at each sample's pixel ``(x, y)``."""
        if self.temp_approx is None:
            raise RuntimeError("Temperature approximation not set. Call set_physics_context() first.")
        
        # Compute temperature from predictions based on physics_mode
        pred_temp = self._compute_predicted_temperature(denorm_pred)
        
        # Get corresponding temperature approximations
        y_idx = spatial_indices[:, 0].long()
        x_idx = spatial_indices[:, 1].long()
        approx_temp = self.temp_approx[y_idx, x_idx]  # (batch_size,)
        
        # RRMSE loss in physical units (Kelvin)
        return self._compute_rrmse(pred_temp, approx_temp)

    def compute_physics_loss(
        self,
        predictions: torch.Tensor,
        spatial_indices: torch.Tensor,
        enable_physics: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Weighted sum of the enabled physics terms, and the value of each term for logging.

        Args:
            predictions: Normalized predictions, ``(batch, 3 n_tau)``.
            spatial_indices: Pixel ``(x, y)`` of each sample, ``(batch, 2)``.
            enable_physics: Gate for all terms at once (used to delay the physics terms by a
                number of epochs); False returns zero.

        Returns:
            ``(total, components)``: the weighted sum as a tensor, and a dict with each
            enabled term's RRMSE (``"wfa"``, ``"doppler"``, ``"temperature"``), the modes and
            the target nodes. Both are empty or zero when the gate is closed or every lambda
            is 0.

        Examples:
            >>> import contextlib, io, tempfile
            >>> import numpy as np, torch
            >>> from utils.normalizer import MhdNormalizer
            >>> rng = np.random.default_rng(0)
            >>> atm = {"T": 5000 + 100 * rng.standard_normal((6, 6, 3)), "Vz": rng.standard_normal((6, 6, 3)),
            ...        "Bz": 100 * rng.standard_normal((6, 6, 3))}
            >>> norm = MhdNormalizer(n_tau=3, raw_values_dir=tempfile.mkdtemp()); norm.update(atm)
            >>> with contextlib.redirect_stdout(io.StringIO()):
            ...     _ = norm.finalize()
            >>> model = PhysicsInformedMSCNN(scales=[1], n_tau_levels=3, lambda_wfa=1.0, lambda_doppler=0.0,
            ...                              lambda_temp=0.0, blos_physics_mode="single_height",
            ...                              blos_target_logtau=-1.0)
            >>> model.set_physics_context(norm, np.array([-2.0, -1.0, 0.0]), blos_approx=np.full((6, 6), 50.0))
            >>> pred = torch.zeros(2, 9, requires_grad=True)
            >>> total, parts = model.compute_physics_loss(pred, torch.tensor([[0, 0], [5, 5]]))
            >>> sorted(k for k in parts if k in ("wfa", "doppler", "temperature")), total.requires_grad
            (['wfa'], True)
        """
        # Return zero if all physics terms are disabled
        # The gate governs ALL physics terms, not just the WFA. It previously applied only
        # to the WFA, which left the ablation arms incomparable: wfa_only trained on plain
        # MSE until the gate opened around epoch 40, while doppler_only and black_body_only
        # had their term active from epoch 1. Same treatment for every arm now.
        if not enable_physics or (
            self.lambda_wfa == 0 and self.lambda_doppler == 0 and self.lambda_temp == 0
        ):
            device = self._get_device()
            return torch.tensor(0.0, device=device), {}
        
        # Denormalize predictions to physical units
        denorm_pred = self._denormalize_predictions(predictions)
        
        device = self._get_device()
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}
        
        # Add physics mode info to components
        loss_components['blos_physics_mode'] = self.blos_physics_mode
        loss_components['vlos_physics_mode'] = self.vlos_physics_mode
        loss_components['temp_physics_mode'] = self.temp_physics_mode
        if self.blos_physics_mode == 'single_height':
            loss_components['blos_target_logtau'] = self.blos_target_logtau
        if self.vlos_physics_mode == 'single_height':
            loss_components['vlos_target_logtau'] = self.vlos_target_logtau
        if self.temp_physics_mode == 'single_height':
            loss_components['temp_target_logtau'] = self.temp_target_logtau
        
        # WFA B_LOS loss (if enabled)
        loss_components['physics_enabled'] = float(enable_physics)
        loss_components['wfa_enabled'] = float(enable_physics and self.lambda_wfa > 0)

        if self.lambda_wfa > 0:
            wfa_loss = self._compute_wfa_loss(denorm_pred, spatial_indices)
            loss_components['wfa'] = wfa_loss.item()
            total_loss = total_loss + self.lambda_wfa * wfa_loss
        
        # Doppler V_LOS loss (if enabled)
        if self.lambda_doppler > 0:
            doppler_loss = self._compute_doppler_loss(denorm_pred, spatial_indices)
            loss_components['doppler'] = doppler_loss.item()
            total_loss = total_loss + self.lambda_doppler * doppler_loss
        
        # Temperature loss (if enabled)
        if self.lambda_temp > 0:
            temp_loss = self._compute_temperature_loss(denorm_pred, spatial_indices)
            loss_components['temperature'] = temp_loss.item()
            total_loss = total_loss + self.lambda_temp * temp_loss
        
        return total_loss, loss_components

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        spatial_indices: Optional[torch.Tensor] = None,
        enable_physics: bool = True,
    ) -> Dict[str, Any]:
        """Total loss: MSE in normalized space over all outputs, plus the physics terms.

        The MSE weights every node of T, Vz and Bz equally. The physics terms are added only
        when ``spatial_indices`` is given, the gate is open and at least one lambda is
        positive.

        Args:
            predictions: Normalized predictions, ``(batch, 3 n_tau)``.
            targets: Normalized targets, same shape.
            spatial_indices: Pixel ``(x, y)`` of each sample, ``(batch, 2)``, or ``None``
                to skip the physics terms.
            enable_physics: Gate for all physics terms.

        Returns:
            Dict with ``"loss"`` (total), ``"mse"``, ``"physics"`` (tensors) and the logging
            components of :meth:`compute_physics_loss`.

        Examples:
            >>> import torch
            >>> m = PhysicsInformedMSCNN(scales=[1], n_tau_levels=1, lambda_wfa=0.0, lambda_doppler=0.0,
            ...                          lambda_temp=0.0)
            >>> out = m.compute_loss(torch.ones(4, 3), torch.zeros(4, 3), torch.zeros(4, 2, dtype=torch.long))
            >>> float(out["loss"]), float(out["mse"]), float(out["physics"])
            (1.0, 1.0, 0.0)
        """
        # Base supervised MSE loss in normalized space
        mse_loss = F.mse_loss(predictions, targets)
        
        # Physics regularization - check if ANY lambda is non-zero and spatial_indices provided
        use_physics = spatial_indices is not None and any([
            enable_physics and self.lambda_wfa > 0,
            enable_physics and self.lambda_doppler > 0,
            enable_physics and self.lambda_temp > 0
        ])
        
        if use_physics:
            physics_loss_total, physics_components = self.compute_physics_loss(
                predictions,
                spatial_indices,
                enable_physics=enable_physics,
            )
        else:
            device = self._get_device()
            physics_loss_total = torch.tensor(0.0, device=device)
            physics_components = {
                'physics_enabled': float(enable_physics),
                'wfa_enabled': float(enable_physics and self.lambda_wfa > 0),
            }

        total_loss = mse_loss + physics_loss_total

        return {
            "loss": total_loss,
            "mse": mse_loss,
            "physics": physics_loss_total,
            **physics_components
        }