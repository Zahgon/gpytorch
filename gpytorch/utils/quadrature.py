#!/usr/bin/env python3

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import Module

from .. import settings


def _pad_with_singletons(obj: torch.Tensor, num_singletons_before: int = 0, num_singletons_after: int = 0):
    """
    Pad obj with singleton dimensions on the left and right

    Example:
        >>> x = torch.randn(10, 5)
        >>> _pad_width_singletons(x, 2, 3).shape
        >>> # [1, 1, 10, 5, 1, 1, 1]
    """
    pass


class GaussHermiteQuadrature1D(Module):
    """
    Implements Gauss-Hermite quadrature for integrating a function with respect to several 1D Gaussian distributions
    in batch mode. Within GPyTorch, this is useful primarily for computing expected log likelihoods for variational
    inference.

    This is implemented as a Module because Gauss-Hermite quadrature has a set of locations and weights that it
    should initialize one time, but that should obey parent calls to .cuda(), .double() etc.
    """

    def __init__(self, num_locs=None):
        super().__init__()
        if num_locs is None:
            num_locs = settings.num_gauss_hermite_locs.value()
        self.num_locs = num_locs

        locations, weights = self._locs_and_weights(num_locs)

        self.locations = locations
        self.weights = weights

    def _apply(self, fn):
        self.locations = fn(self.locations)
        self.weights = fn(self.weights)
        return super()._apply(fn)

    def _locs_and_weights(self, num_locs):
        """
        Get locations and weights for Gauss-Hermite quadrature. Note that this is **not** intended to be used
        externally, because it directly creates tensors with no knowledge of a device or dtype to cast to.

        Instead, create a GaussHermiteQuadrature1D object and get the locations and weights from buffers.
        """
        locations, weights = np.polynomial.hermite.hermgauss(num_locs)
        locations = torch.Tensor(locations)
        weights = torch.Tensor(weights)
        return locations, weights

    def forward(self, func, gaussian_dists):
        """
        Runs Gauss-Hermite quadrature on the callable func, integrating against the Gaussian distributions specified
        by gaussian_dists.

        Args:
            - func (callable): Function to integrate
            - gaussian_dists (Distribution): Either a MultivariateNormal whose covariance is assumed to be diagonal
                or a :obj:`torch.distributions.Normal`.
        Returns:
            - Result of integrating func against each univariate Gaussian in gaussian_dists.
        """
        pass
