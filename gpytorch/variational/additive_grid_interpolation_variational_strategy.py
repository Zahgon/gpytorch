from __future__ import annotations

from collections.abc import Iterable

import torch
from linear_operator.operators import LinearOperator
from torch import LongTensor, Tensor

from ..distributions import Delta, MultivariateNormal
from ..models import ApproximateGP
from ..variational._variational_distribution import _VariationalDistribution
from ..variational.grid_interpolation_variational_strategy import GridInterpolationVariationalStrategy


class AdditiveGridInterpolationVariationalStrategy(GridInterpolationVariationalStrategy):
    def __init__(
        self,
        model: ApproximateGP,
        grid_size: int,
        grid_bounds: Iterable[tuple[float, float]],
        num_dim: int,
        variational_distribution: _VariationalDistribution,
        mixing_params: bool = False,
        sum_output: bool = True,
    ):
        super().__init__(model, grid_size, grid_bounds, variational_distribution)
        self.num_dim = num_dim
        self.sum_output = sum_output
        # Mixing parameters
        if mixing_params:
            self.register_parameter(name="mixing_params", parameter=torch.nn.Parameter(torch.ones(num_dim) / num_dim))

    @property
    def prior_distribution(self) -> MultivariateNormal:
        # If desired, models can compare the input to forward to inducing_points and use a GridKernel for space
        # efficiency.
        # However, when using a default VariationalDistribution which has an O(m^2) space complexity anyways,
        # we find that GridKernel is typically not worth it due to the moderate slow down of using FFTs.
        pass

    def _compute_grid(self, inputs: Tensor) -> tuple[LongTensor, Tensor]:
        pass

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None = None,
        diag: bool = True,
        **kwargs,
    ) -> MultivariateNormal:
        pass
