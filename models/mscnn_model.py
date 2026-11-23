"""Multi-scale 1-D CNN (MSCNN) that maps Stokes profiles to atmospheric stratifications.

The input is ``(batch, in_channels, n_wl)``: normalized Stokes parameters (I and V today)
on the 112-point Hinode grid. Each scale ``s`` averages groups of ``s`` wavelength samples
(:class:`CoarseGrain`), then the same convolution block (one Conv1d, ReLU and max-pool,
shared by all scales) extracts features. The feature maps of all scales are concatenated
along the wavelength axis, flattened and passed through a stack of dense layers. The
output is ``(batch, 3 n_tau)``: normalized T, then Vz, then Bz, each ordered by node.

With the defaults (scales 1, 2, 3; 16 filters; kernel 5; pool 2; 112 points) the
flattened size is 16 x (54 + 26 + 16) = 1536 and the model has 9.65 M parameters for
45 nodes, almost all in the four 1536 x 1536 dense layers.

Examples:
    >>> import torch
    >>> model = MSCNNInversionModel(scales=[1, 2, 3], output_features=135)
    >>> model(torch.zeros(4, 2, 112)).shape
    torch.Size([4, 135])
"""

import torch
import torch.nn as nn
import numpy as np


def _conv1d_output_length(input_size: int, kernel_size: int, stride: int = 1, padding: int = 0) -> int:
	"""Output length of a Conv1d: ``(L - k + 2 p) // stride + 1``.

	Args:
	    input_size: Input length L.
	    kernel_size: Kernel size k.
	    stride: Stride.
	    padding: Padding p on each side.

	Returns:
	    The output length.

	Examples:
	    >>> _conv1d_output_length(112, 5)
	    108
	"""
	return (input_size - kernel_size + 2 * padding) // stride + 1


class CoarseGrain(nn.Module):
	"""Average non-overlapping groups of ``scale`` samples along the last axis.

	Trailing samples that do not fill a group are dropped, so 112 points become 56 at
	scale 2 and 37 at scale 3 (the reddest point is not used at that scale).

	Examples:
	    >>> import torch
	    >>> CoarseGrain(3)(torch.arange(7.0).reshape(1, 1, 7))
	    tensor([[[1., 4.]]])
	"""
	def __init__(self, scale: int) -> None:
		"""Store the averaging factor.

		Args:
		    scale: Number of samples averaged into one.
		"""
		super().__init__()
		self.scale = scale

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Average groups of ``scale`` samples along the last axis, dropping the remainder.

		Args:
		    x: ``(batch, channels, L)``.

		Returns:
		    ``(batch, channels, L // scale)``.

		Examples:
		    >>> import torch
		    >>> CoarseGrain(2)(torch.zeros(1, 2, 112)).shape
		    torch.Size([1, 2, 56])
		"""
		unfolded = x.unfold(dimension=2, size=self.scale, step=self.scale)
		return unfolded.mean(dim=-1)


class ConvBlock(nn.Module):
	"""One Conv1d, ReLU and MaxPool1d stage, shared by every scale of :class:`MultiScaleFeatureMapping`.

	A second stage (``c2_filters``) exists in the code but is disabled; ``c2_filters`` is
	accepted and ignored.
	"""
	def __init__(
		self,
		*,
		in_channels: int = 2,
		c1_filters: int = 16,
		c2_filters: int = 32,
		kernel_size: int = 5,
		stride: int = 1,
		padding: int = 0,
		pool_size: int = 2,
	) -> None:
		"""Build the convolution stage.

		Args:
		    in_channels: Number of Stokes channels in the input (2 for I and V).
		    c1_filters: Number of convolution filters.
		    c2_filters: Filters of the disabled second stage (ignored).
		    kernel_size: Convolution kernel size in samples.
		    stride: Convolution stride.
		    padding: Convolution padding.
		    pool_size: Max-pool size (also its stride).
		"""
		super().__init__()
		self.c1 = nn.Sequential(
			nn.Conv1d(
				in_channels=in_channels,
				out_channels=c1_filters,
				kernel_size=kernel_size,
				stride=stride,
				padding=padding,
			),
			nn.ReLU(),
			nn.MaxPool1d(kernel_size=pool_size),
		)

		# Second conv block disabled for ablation (use only first conv layer)
		# self.c2 = nn.Sequential(
		# 	nn.Conv1d(
		# 		in_channels=c1_filters,
		# 		out_channels=c2_filters,
		# 		kernel_size=kernel_size,
		# 		stride=stride,
		# 		padding=padding,
		# 	),
		# 	nn.ReLU(),
		# 	nn.MaxPool1d(kernel_size=pool_size),
		# )

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Convolve, rectify and pool.

		Args:
		    x: ``(batch, in_channels, L)``.

		Returns:
		    ``(batch, c1_filters, (L - kernel_size + 1) // pool_size)`` without padding.

		Examples:
		    >>> import torch
		    >>> ConvBlock()(torch.zeros(1, 2, 112)).shape
		    torch.Size([1, 16, 54])
		"""
		x = self.c1(x)
		# x = self.c2(x)
		return x


class MultiScaleFeatureMapping(nn.Module):
	"""Run the shared :class:`ConvBlock` on each coarse-grained copy of the input and concatenate.

	The output is ``(batch, c1_filters, sum over scales of the pooled lengths)``.
	"""
	def __init__(
		self,
		*,
		scales: list[int],
		in_channels: int = 2,
		c1_filters: int = 16,
		c2_filters: int = 32,
		kernel_size: int = 5,
		stride: int = 1,
		padding: int = 0,
		pool_size: int = 2,
	) -> None:
		"""Build one coarse-graining per scale and the shared convolution stage.

		Args:
		    scales: Coarse-graining factors, e.g. ``[1, 2, 3]``.
		    in_channels: Number of Stokes channels.
		    c1_filters: Number of convolution filters.
		    c2_filters: Filters of the disabled second stage (ignored).
		    kernel_size: Convolution kernel size in samples.
		    stride: Convolution stride.
		    padding: Convolution padding.
		    pool_size: Max-pool size.
		"""
		super().__init__()
		self.scales = scales
		self.out_filters = c1_filters  # c2 disabled; output channels come from c1
		self.coarse_grains = nn.ModuleDict({f"scale_{s}": CoarseGrain(scale=s) for s in scales})
		self.conv_block = ConvBlock(
			in_channels=in_channels,
			c1_filters=c1_filters,
			c2_filters=c2_filters,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			pool_size=pool_size,
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Concatenate the pooled feature maps of every scale along the wavelength axis.

		Args:
		    x: ``(batch, in_channels, L)``.

		Returns:
		    ``(batch, c1_filters, total length)``; for L = 112 and scales 1, 2, 3 the total
		    length is 54 + 26 + 16 = 96.

		Examples:
		    >>> import torch
		    >>> MultiScaleFeatureMapping(scales=[1, 2, 3])(torch.zeros(1, 2, 112)).shape
		    torch.Size([1, 16, 96])
		"""
		feature_maps = torch.empty((x.size(0), self.out_filters, 0), dtype=torch.float32, device=x.device)
		for s in self.scales:
			stokes_cg = self.coarse_grains[f"scale_{s}"](x)
			conv_block_output = self.conv_block(stokes_cg)
			feature_maps = torch.cat((feature_maps, conv_block_output), dim=-1)
		return feature_maps


class LinearBlock(nn.Module):
    """``n_layers`` square dense layers with ReLU, then a linear output layer (no activation)."""
    def __init__(
        self, 
        *, 
        in_features: int, 
        out_features: int, 
        n_layers: int = 4
    ) -> None:
        """Build the hidden and output layers.

        Args:
            in_features: Width of the input and of every hidden layer.
            out_features: Number of outputs.
            n_layers: Number of hidden layers.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_features=in_features, out_features=in_features),
                    nn.ReLU(),
                )
                for _ in range(n_layers)
            ]
        )
        self.output_layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the hidden layers, then the output layer.

        Args:
            x: ``(batch, in_features)``.

        Returns:
            ``(batch, out_features)``.

        Examples:
            >>> import torch
            >>> LinearBlock(in_features=8, out_features=3, n_layers=2)(torch.zeros(5, 8)).shape
            torch.Size([5, 3])
        """
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class MSCNNInversionModel(nn.Module):
    """The MSCNN inversion model: multi-scale features, flatten, dense layers.

    The flattened size is computed from ``input_length`` in the constructor, so the model
    only accepts inputs of that length.
    """
    def __init__(
        self,
        *,
        scales: list[int],
        in_channels: int = 2,
        c1_filters: int = 16,
        c2_filters: int = 32,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = 0,
        pool_size: int = 2,
        n_linear_layers: int = 4,
        output_features: int | None = None,
        input_length: int = 112,
        n_tau_levels: int = 21,
    ) -> None:
        """Build the network for a fixed input length.

        Args:
            scales: Coarse-graining factors, e.g. ``[1, 2, 3]``.
            in_channels: Number of Stokes channels in the input.
            c1_filters: Number of convolution filters.
            c2_filters: Filters of the disabled second stage (ignored).
            kernel_size: Convolution kernel size in samples.
            stride: Convolution stride.
            padding: Convolution padding.
            pool_size: Max-pool size.
            n_linear_layers: Number of hidden dense layers.
            output_features: Number of outputs (3 n_tau). ``None`` uses ``3 n_tau_levels``.
            input_length: Number of wavelength samples (112 for Hinode/SP).
            n_tau_levels: Number of nodes, used only when ``output_features`` is ``None``.

        Raises:
            ValueError: If the configuration leaves no features to flatten.
        """
        super().__init__()

        self.msfm = MultiScaleFeatureMapping(
            scales=scales,
            in_channels=in_channels,
            c1_filters=c1_filters,
            c2_filters=c2_filters,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            pool_size=pool_size,
        )

        # Compute flatten size from expected input length.
        # Second conv block disabled: only one conv+pool stage.
        total_features = 0
        for s in scales:
            length = input_length // s
            length = _conv1d_output_length(length, kernel_size, stride=stride, padding=padding)
            length = length // pool_size
            total_features += length
        flatten_size = total_features * c1_filters
        if flatten_size <= 0:
            raise ValueError(
                f"Invalid flatten_size={flatten_size}. Check input_length/scales/kernel/pool configuration."
            )

        if output_features is None:
            output_features = 3 * int(n_tau_levels)

        self.flatten = nn.Flatten()
        self.linear_block = LinearBlock(
            in_features=flatten_size,
            out_features=int(output_features),
            n_layers=n_linear_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized Stokes profiles to normalized stratifications.

        Args:
            x: ``(batch, in_channels, input_length)``.

        Returns:
            ``(batch, output_features)``: T, then Vz, then Bz, each ordered by node.

        Examples:
            >>> import torch
            >>> MSCNNInversionModel(scales=[1, 2], n_tau_levels=4)(torch.zeros(1, 2, 112)).shape
            torch.Size([1, 12])
        """
        x = self.msfm(x)
        x = self.flatten(x)
        return self.linear_block(x)


__all__ = [
	"CoarseGrain",
	"ConvBlock",
	"MultiScaleFeatureMapping",
	"LinearBlock",
	"MSCNNInversionModel",
]

"""
import torch
from models.mscnn_model import MSCNNInversionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate with your preferred scales and expected input length
model = MSCNNInversionModel(scales=[1, 2, 3], input_length=112).to(device)

# Dummy batch of Stokes profiles: shape (batch, channels=2, length=112)
dummy_input = torch.randn(4, 2, 112, device=device)

with torch.no_grad():
    output = model(dummy_input)

print("Output shape:", output.shape)  # should be (4, 63) for 3*21 outputs
"""