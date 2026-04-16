#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    PsdSumLinearOperator,
    RootLinearOperator,
    TriangularLinearOperator,
    ZeroLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from .. import settings
from ..distributions import MultivariateNormal
from ..utils.memoize import add_to_cache, cached
from ._variational_strategy import _VariationalStrategy
from .cholesky_variational_distribution import CholeskyVariationalDistribution


class UnwhitenedVariationalStrategy(_VariationalStrategy):
    r"""
    Similar to :obj:`~gpytorch.variational.VariationalStrategy`, but does not perform the
    whitening operation. In almost all cases :obj:`~gpytorch.variational.VariationalStrategy`
    is preferable, with a few exceptions:

    - When the inducing points are exactly equal to the training points (i.e. :math:`\mathbf Z = \mathbf X`).
      Unwhitened models are faster in this case.

    - When the number of inducing points is very large (e.g. >2000). Unwhitened models can use CG for faster
      computation.

    :param ~model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability
    """

    has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        # Maybe used - if we're not using CG
        pass

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        pass

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> tuple[Tensor, Tensor]:
        # TODO: implement for other distributions
        # retrieve the variational mean, m and covariance matrix, S.
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
        # If our points equal the inducing points, we're done
        pass
