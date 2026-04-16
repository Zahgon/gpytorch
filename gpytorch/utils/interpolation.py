#!/usr/bin/env python3

from __future__ import annotations

import warnings
from functools import reduce
from operator import mul

import torch
from linear_operator.utils.interpolation import left_interp as _left_interp, left_t_interp as _left_t_interp

from .grid import convert_legacy_grid


class Interpolation:
    def _cubic_interpolation_kernel(self, scaled_grid_dist):
        """
        Computes the interpolation kernel u() for points X given the scaled
        grid distances:
                                    (X-x_{t})/s
        where s is the distance between neighboring grid points. Note that,
        in this context, the word "kernel" is not used to mean a covariance
        function as in the rest of the package. For more details, see the
        original paper Keys et al., 1989, equation (4).

        scaled_grid_dist should be an n-by-g matrix of distances, where the
        (ij)th element is the distance between the ith data point in X and the
        jth element in the grid.

        Note that, although this method ultimately expects a scaled distance matrix,
        it is only intended to be used on single dimensional data.
        """
        pass

    def interpolate(self, x_grid: list[torch.Tensor], x_target: torch.Tensor, interp_points=range(-2, 2), eps=1e-10):
        pass


def left_interp(interp_indices: torch.LongTensor, interp_values: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    warnings.warn(
        "gpytorch.utils.interpolation.left_interp is deprecated in favor of "
        "linear_operator.utils.interpolation.left_interp.",
        DeprecationWarning,
    )
    return _left_interp(interp_indices=interp_indices, interp_values=interp_values, rhs=rhs)


def left_t_interp(
    interp_indices: torch.LongTensor, interp_values: torch.Tensor, rhs: torch.Tensor, output_dim: int
) -> torch.Tensor:
    warnings.warn(
        "gpytorch.utils.interpolation.left_t_interp is deprecated in favor of "
        "linear_operator.utils.interpolation.left_t_interp.",
        DeprecationWarning,
    )
    return _left_t_interp(interp_indices=interp_indices, interp_values=interp_values, rhs=rhs, output_dim=output_dim)
