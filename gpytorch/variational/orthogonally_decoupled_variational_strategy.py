#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import LinearOperator
from torch import Tensor

from ..distributions import MultivariateNormal
from ..utils.memoize import add_to_cache, cached
from ._variational_distribution import _VariationalDistribution
from ._variational_strategy import _VariationalStrategy
from .delta_variational_distribution import DeltaVariationalDistribution


class OrthogonallyDecoupledVariationalStrategy(_VariationalStrategy):
    r"""
    Implements orthogonally decoupled VGPs as defined in `Salimbeni et al. (2018)`_.
    This variational strategy uses a different set of inducing points for the mean and covariance functions.
    The idea is to use more inducing points for the (computationally efficient) mean and fewer inducing points for the
    (computationally expensive) covaraince.

    This variational strategy defines the inducing points/:obj:`~gpytorch.variational._VariationalDistribution`
    for the mean function.
    It then wraps a different :obj:`~gpytorch.variational._VariationalStrategy` which
    defines the covariance inducing points.

    :param covar_variational_strategy:
        The variational strategy for the covariance term.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    Example:
        >>> mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>> covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
        >>>
        >>> covar_variational_strategy = gpytorch.variational.VariationalStrategy(
        >>>     model, covar_inducing_points,
        >>>     gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
        >>>     learn_inducing_locations=True
        >>> )
        >>>
        >>> variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
        >>>     covar_variational_strategy, mean_inducing_points,
        >>>     gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
        >>> )

    .. _Salimbeni et al. (2018):
        https://arxiv.org/abs/1809.08820
    """

    def __init__(
        self,
        covar_variational_strategy: _VariationalStrategy,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        jitter_val: float | None = None,
    ):
        if not isinstance(variational_distribution, DeltaVariationalDistribution):
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        super().__init__(
            covar_variational_strategy,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
            jitter_val=jitter_val,
        )
        self.base_variational_strategy = covar_variational_strategy

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
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

    def kl_divergence(self) -> Tensor:
        mean = self.variational_distribution.mean
        induc_induc_covar = self.prior_distribution.lazy_covariance_matrix
        kl = self.model.kl_divergence() + ((induc_induc_covar @ mean.unsqueeze(-1)).squeeze(-1) * mean).sum(-1).mul(0.5)
        return kl
