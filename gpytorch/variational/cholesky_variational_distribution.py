#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator

from ..distributions import MultivariateNormal
from ._variational_distribution import _VariationalDistribution


class CholeskyVariationalDistribution(_VariationalDistribution):
    """
    A :obj:`~gpytorch.variational._VariationalDistribution` that is defined to be a multivariate normal distribution
    with a full covariance matrix.

    The most common way this distribution is defined is to parameterize it in terms of a mean vector and a covariance
    matrix. In order to ensure that the covariance matrix remains positive definite, we only consider the lower
    triangle.

    :param num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :param mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(
        self,
        num_inducing_points: int,
        batch_shape: torch.Size = torch.Size([]),
        mean_init_std: float = 1e-3,
        **kwargs,
    ):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", parameter=torch.nn.Parameter(covar_init))

    def forward(self) -> MultivariateNormal:
        pass

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        pass
