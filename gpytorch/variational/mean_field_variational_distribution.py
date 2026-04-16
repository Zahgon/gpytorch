#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import DiagLinearOperator

from ..distributions import MultivariateNormal
from ._variational_distribution import _VariationalDistribution


class MeanFieldVariationalDistribution(_VariationalDistribution):
    """
    A :obj:`~gpytorch.variational._VariationalDistribution` that is defined to be a multivariate normal distribution
    with a diagonal covariance matrix. This will not be as flexible/expressive as a
    :obj:`~gpytorch.variational.CholeskyVariationalDistribution`.

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :type batch_shape: :obj:`torch.Size`, optional
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(self, num_inducing_points, batch_shape=torch.Size([]), mean_init_std=1e-3, **kwargs):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.ones(num_inducing_points)
        mean_init = mean_init.repeat(*batch_shape, 1)
        covar_init = covar_init.repeat(*batch_shape, 1)

        self.register_parameter(name="variational_mean", parameter=torch.nn.Parameter(mean_init))
        self.register_parameter(name="_variational_stddev", parameter=torch.nn.Parameter(covar_init))

    @property
    def variational_stddev(self):
        # TODO: if we don't multiply self._variational_stddev by a mask of one, Pyro models fail
        # not sure where this bug is occuring (in Pyro or PyTorch)
        # throwing this in as a hotfix for now - we should investigate later
        pass

    def forward(self):
        # TODO: if we don't multiply self._variational_stddev by a mask of one, Pyro models fail
        # not sure where this bug is occuring (in Pyro or PyTorch)
        # throwing this in as a hotfix for now - we should investigate later
        pass

    def initialize_variational_distribution(self, prior_dist):
        pass
