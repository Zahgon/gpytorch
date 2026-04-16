#!/usr/bin/env python3

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from gpytorch import settings

from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution

from ..distributions import MultivariateNormal
from ..models import ApproximateGP
from ..settings import _linalg_dtype_cholesky, trace_mode
from ..utils.errors import CachingError
from ..utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from ..utils.warnings import OldVersionWarning
from . import _VariationalDistribution


def _ensure_updated_strategy_flag_set(
    state_dict: dict[str, Tensor],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: Iterable[str],
    unexpected_keys: Iterable[str],
    error_msgs: Iterable[str],
):
    pass


class ComputePredictiveUpdates(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        chol: Tensor,
        induc_data_covar: Tensor,
        middle: Tensor,
        inducing_values: Tensor,
    ) -> tuple[Tensor, Tensor]:
        r"""Compute the predictive mean and variance updates as in `VariationalStrategy._compute_predictive_updates`.

        This function doesn't compute the updates to the off-diagonal entries in the predictive covariance. Only the
        variance update is computed.
        """
        pass

    @staticmethod
    def backward(
        ctx,
        d_mean: Tensor,
        d_variance: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        r"""A custom backward pass more efficient than PyTorch's autograd by rearranging tensor operations.

        This backward is bottlenecked by two O(m^2 n) matmuls whre `m` is the number of inducing points and `n` is the
        number of data points. In contrast, PyTorch's backward pass would require three O(m^2 n) matmuls and a O(m^2 n)
        triangular solve. Thus, this implementation is about 2x faster when `m << n`.
        """
        pass


class VariationalStrategy(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: float | None = None,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        pass

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        pass

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        pass

    def _compute_predictive_updates(
        self,
        chol: LinearOperator,
        induc_data_covar: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None,
        prior_covar: LinearOperator,
        diag: bool = True,
    ) -> tuple[Tensor, LinearOperator]:
        r"""Compute the predictive mean and covariance updates. Adding the return values of this method to the prior
        mean and covariance yields the predictive mean and covariance.

        The predictive mean update is `K_{XZ} K_{ZZ}^{-1/2} m`.

        The predictive covariance update is `K_{XZ} K_{ZZ}^{-1/2} (S - I) K_{ZZ}^{-1/2} K_{ZX}`.

        :param chol: The Cholesky factor `K_{ZZ}^{-1/2}`.
        :param induc_data_covar: The covariance between the inducing points and the data `K_{ZX}`.
        :param inducing_values: The whitened variational mean `m`.
        :param variational_inducing_covar: The variational covariance `S`.
        :param prior_covar: The prior covariance, typically an identity matrix `I`.
        :param diag: If true, this method computes the predictive variance instead of the full covariance in train mode
            if there are more data than inducing points.
        :return: The predictive mean update and the predictive covariance update.
        """
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
        # Compute full prior distribution
        pass

    def __call__(self, x: Tensor, prior: bool = False, diag: bool = True, **kwargs) -> MultivariateNormal:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u). Whitening needs the full covariance.
                prior_function_dist = self(self.inducing_points, prior=True, diag=False)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultivariateNormal):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, diag=diag, **kwargs)
