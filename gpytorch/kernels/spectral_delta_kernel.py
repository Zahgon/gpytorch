from __future__ import annotations

import math

import torch
from linear_operator.operators import MatmulLinearOperator, RootLinearOperator

from ..constraints import Interval, Positive
from .kernel import Kernel


class SpectralDeltaKernel(Kernel):
    """
    A kernel that supports spectral learning for GPs, where the underlying spectral density is modeled as a mixture
    of delta distributions (e.g., with point masses). This has been explored e.g. in Lazaro-Gredilla et al., 2010.

    Conceptually, this kernel is similar to random Fourier features as implemented in RFFKernel, but instead of sampling
    a Gaussian to determine the spectrum sites, they are treated as learnable parameters.

    When using CG for inference, this kernel supports linear space and time (in N) for training and inference.

    :param int num_dims: Dimensionality of input data that this kernel will operate on. Note that if active_dims is
        used, this should be the length of the active dim set.
    :param int num_deltas: Number of point masses to learn.
    """

    has_lengthscale = True

    def __init__(
        self,
        num_dims: int,
        num_deltas: int | None = 128,
        Z_constraint: Interval | None = None,
        batch_shape: torch.Size | None = torch.Size([]),
        **kwargs,
    ):
        Kernel.__init__(self, has_lengthscale=True, batch_shape=batch_shape, **kwargs)

        self.raw_Z = torch.nn.Parameter(torch.rand(*batch_shape, num_deltas, num_dims))

        if Z_constraint:
            self.register_constraint("raw_Z", Z_constraint)
        else:
            self.register_constraint("raw_Z", Positive())

        self.num_dims = num_dims

    def initialize_from_data(self, train_x, train_y):
        """
        Initialize the point masses for this kernel from the empirical spectrum of the data. To do this, we estimate
        the empirical spectrum's CDF and then simply sample from it. This is analogous to how the SM kernel's mixture
        is initialized, but we skip the last step of fitting a GMM to the samples and just use the samples directly.
        """
        import numpy as np

        N = train_x.size(-2)
        # Use torch.fft instead of scipy for FFT (stay in PyTorch land)
        train_y_np = train_y.cpu().detach()
        emp_spect = torch.abs(torch.fft.fft(train_y_np)) ** 2 / N
        M = math.floor(N / 2)

        freq1 = torch.arange(M + 1, dtype=train_y_np.dtype)
        freq2 = torch.arange(-M + 1, 0, dtype=train_y_np.dtype)
        freq = torch.hstack((freq1, freq2)) / N
        freq = freq[: M + 1]
        emp_spect = emp_spect[: M + 1]

        # Use torch.trapezoid (already in PyTorch)
        total_area = torch.trapezoid(emp_spect, freq).item()
        spec_cdf = torch.cat([torch.zeros(1), torch.cumulative_trapezoid(emp_spect, freq)]).numpy()
        spec_cdf = spec_cdf / total_area
        freq_np = freq.numpy()

        a = np.random.rand(self.raw_Z.size(-2), 1)
        p, q = np.histogram(a, spec_cdf)
        bins = np.digitize(a, q)
        slopes = (spec_cdf[bins] - spec_cdf[bins - 1]) / (freq_np[bins] - freq_np[bins - 1])
        intercepts = spec_cdf[bins - 1] - slopes * freq_np[bins - 1]
        inv_spec = (a - intercepts) / slopes

        self.Z = inv_spec

    def initialize_from_data_simple(self, train_x, train_y, **kwargs):
        pass

    @property
    def Z(self):
        pass

    @Z.setter
    def Z(self, value):
        pass

    def _set_Z(self, value):
        pass

    def forward(self, x1, x2, diag=False, **params):
        pass
