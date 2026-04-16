#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import InterpolatedLinearOperator

from ..models.exact_prediction_strategies import InterpolatedPredictionStrategy
from ..utils.grid import create_grid
from ..utils.interpolation import Interpolation
from .grid_kernel import GridKernel
from .kernel import Kernel


class GridInterpolationKernel(GridKernel):
    r"""
    Implements the KISS-GP (or SKI) approximation for a given kernel.
    It was proposed in `Kernel Interpolation for Scalable Structured Gaussian Processes`_,
    and offers extremely fast and accurate Kernel approximations for large datasets.

    Given a base kernel `k`, the covariance :math:`k(\mathbf{x_1}, \mathbf{x_2})` is approximated by
    using a grid of regularly spaced *inducing points*:

    .. math::

       \begin{equation*}
          k(\mathbf{x_1}, \mathbf{x_2}) = \mathbf{w_{x_1}}^\top K_{U,U} \mathbf{w_{x_2}}
       \end{equation*}

    where

    * :math:`U` is the set of gridded inducing points

    * :math:`K_{U,U}` is the kernel matrix between the inducing points

    * :math:`\mathbf{w_{x_1}}` and :math:`\mathbf{w_{x_2}}` are sparse vectors based on
      :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}` that apply cubic interpolation.

    The user should supply the size of the grid (using the grid_size attribute).
    To choose a reasonable grid value, we highly recommend using the
    :func:`gpytorch.utils.grid.choose_grid_size` helper function.
    The bounds of the grid will automatically be determined by data.

    (Alternatively, you can hard-code bounds using the grid_bounds, which
    will speed up this kernel's computations.)

    .. note::

        `GridInterpolationKernel` can only wrap **stationary kernels** (such as RBF, Matern,
        Periodic, Spectral Mixture, etc.)

    Args:
        base_kernel (Kernel):
            The kernel to approximate with KISS-GP
        grid_size (Union[int, List[int]]):
            The size of the grid in each dimension.
            If a single int is provided, then every dimension will have the same grid size.
        num_dims (int):
            The dimension of the input data. Required if `grid_bounds=None`
        grid_bounds (tuple(float, float), optional):
            The bounds of the grid, if known (high performance mode).
            The length of the tuple must match the number of dimensions.
            The entries represent the min/max values for each dimension.
        active_dims (tuple of ints, optional):
            Passed down to the `base_kernel`.

    .. _Kernel Interpolation for Scalable Structured Gaussian Processes:
        http://proceedings.mlr.press/v37/wilson15.pdf
    """

    def __init__(
        self,
        base_kernel: Kernel,
        grid_size: int | list[int],
        num_dims: int | None = None,
        grid_bounds: tuple[float, float] | None = None,
        active_dims: tuple[int, ...] | None = None,
    ):
        has_initialized_grid = 0
        grid_is_dynamic = True

        # Make some temporary grid bounds, if none exist
        if grid_bounds is None:
            if num_dims is None:
                raise RuntimeError("num_dims must be supplied if grid_bounds is None")
            else:
                # Create some temporary grid bounds - they'll be changed soon
                grid_bounds = tuple((-1.0, 1.0) for _ in range(num_dims))
        else:
            has_initialized_grid = 1
            grid_is_dynamic = False
            if num_dims is None:
                num_dims = len(grid_bounds)
            elif num_dims != len(grid_bounds):
                raise RuntimeError(
                    "num_dims ({}) disagrees with the number of supplied "
                    "grid_bounds ({})".format(num_dims, len(grid_bounds))
                )

        if isinstance(grid_size, int):
            grid_sizes = [grid_size for _ in range(num_dims)]
        else:
            grid_sizes = list(grid_size)

        if len(grid_sizes) != num_dims:
            raise RuntimeError("The number of grid sizes provided through grid_size do not match num_dims.")

        # Initialize values and the grid
        self.grid_is_dynamic = grid_is_dynamic
        self.num_dims = num_dims
        self.grid_sizes = grid_sizes
        self.grid_bounds = grid_bounds
        grid = create_grid(self.grid_sizes, self.grid_bounds)

        super().__init__(
            base_kernel=base_kernel,
            grid=grid,
            interpolation_mode=True,
            active_dims=active_dims,
        )
        self.register_buffer("has_initialized_grid", torch.tensor(has_initialized_grid, dtype=torch.bool))

    @property
    def _tight_grid_bounds(self):
        pass

    def _compute_grid(self, inputs, last_dim_is_batch=False):
        pass

    def _inducing_forward(self, last_dim_is_batch, **params):
        pass

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # See if we need to update the grid or not
        pass

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        return InterpolatedPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
