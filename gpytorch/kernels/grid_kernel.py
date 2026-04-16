#!/usr/bin/env python3

from __future__ import annotations

import warnings

import torch
from linear_operator import to_dense
from linear_operator.operators import KroneckerProductLinearOperator, ToeplitzLinearOperator
from torch import Tensor

from .. import settings
from ..utils.grid import convert_legacy_grid, create_data_from_grid
from .kernel import Kernel


class GridKernel(Kernel):
    r"""
    If the input data :math:`X` are regularly spaced on a grid, then
    `GridKernel` can dramatically speed up computatations for stationary kernel.

    GridKernel exploits Toeplitz and Kronecker structure within the covariance matrix.
    See `Fast kernel learning for multidimensional pattern extrapolation`_ for more info.

    .. note::

        `GridKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        base_kernel (Kernel):
            The kernel to speed up with grid methods.
        grid (Tensor):
            A g x d tensor where column i consists of the projections of the
            grid in dimension i.
        active_dims (tuple of ints, optional):
            Passed down to the `base_kernel`.
        interpolation_mode (bool):
            Used for GridInterpolationKernel where we want the covariance
            between points in the projections of the grid of each dimension.
            We do this by treating `grid` as d batches of g x 1 tensors by
            calling base_kernel(grid, grid) with last_dim_is_batch to get a d x g x g Tensor
            which we Kronecker product to get a g x g KroneckerProductLinearOperator.

    .. _Fast kernel learning for multidimensional pattern extrapolation:
        http://www.cs.cmu.edu/~andrewgw/manet.pdf
    """

    is_stationary = True

    def __init__(
        self,
        base_kernel: Kernel,
        grid: Tensor,
        interpolation_mode: bool | None = False,
        active_dims: bool | None = None,
    ):
        if not base_kernel.is_stationary:
            raise RuntimeError("The base_kernel for GridKernel must be stationary.")

        super().__init__(active_dims=active_dims)
        if torch.is_tensor(grid):
            grid = convert_legacy_grid(grid)
        self.interpolation_mode = interpolation_mode
        self.base_kernel = base_kernel
        self.num_dims = len(grid)
        self.register_buffer_list("grid", grid)
        if not self.interpolation_mode:
            self.register_buffer("full_grid", create_data_from_grid(grid))

    def _clear_cache(self):
        pass

    def register_buffer_list(self, base_name, tensors):
        """Helper to register several buffers at once under a single base name"""
        for i, tensor in enumerate(tensors):
            self.register_buffer(base_name + "_" + str(i), tensor)

    @property
    def grid(self):
        return [getattr(self, f"grid_{i}") for i in range(self.num_dims)]

    def update_grid(self, grid):
        """
        Supply a new `grid` if it ever changes.
        """
        pass

    @property
    def is_ragged(self):
        pass

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        pass

    def num_outputs_per_input(self, x1, x2):
        pass
