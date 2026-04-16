#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator import LinearOperator
from linear_operator.operators import InterpolatedLinearOperator
from linear_operator.utils.interpolation import left_interp
from torch import Tensor

from ..distributions import MultivariateNormal
from ..utils.interpolation import Interpolation
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy


class GridInterpolationVariationalStrategy(_VariationalStrategy):
    r"""
    This strategy constrains the inducing points to a grid and applies a deterministic
    relationship between :math:`\mathbf f` and :math:`\mathbf u`.
    It was introduced by `Wilson et al. (2016)`_.

    Here, the inducing points are not learned. Instead, the strategy
    automatically creates inducing points based on a set of grid sizes and grid
    bounds.

    .. _Wilson et al. (2016):
        https://arxiv.org/abs/1611.00336

    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param int grid_size: Size of the grid
    :param list grid_bounds: Bounds of each dimension of the grid (should be a list of (float, float) tuples)
    :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    """

    def __init__(self, model, grid_size, grid_bounds, variational_distribution):
        grid = torch.zeros(grid_size, len(grid_bounds))
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size**i : (j + 1) * grid_size**i, i].fill_(grid[j, i])
                if prev_points is not None:
                    inducing_points[j * grid_size**i : (j + 1) * grid_size**i, :i].copy_(prev_points)
            prev_points = inducing_points[: grid_size ** (i + 1), : (i + 1)]

        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations=False)
        object.__setattr__(self, "model", model)

        self.register_buffer("grid", grid)

    def _compute_grid(self, inputs):
        pass

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        pass

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None = None,
        diag: bool = True,
        **kwargs,
    ):
        pass
