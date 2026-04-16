#!/usr/bin/env python3

from __future__ import annotations

import functools
from abc import ABC, abstractproperty
from copy import deepcopy

import torch
from linear_operator.operators import LinearOperator
from torch import Tensor

from .. import settings
from ..distributions import Delta, Distribution, MultivariateNormal
from ..kernels import Kernel
from ..likelihoods import GaussianLikelihood
from ..means import Mean
from ..models import ApproximateGP, ExactGP
from ..models.exact_prediction_strategies import DefaultPredictionStrategy
from ..module import Module
from ..utils.memoize import add_to_cache, cached, clear_cache_hook
from . import _VariationalDistribution


class _BaseExactGP(ExactGP):
    def __init__(
        self,
        train_inputs: Tensor | tuple[Tensor, ...] | None,
        train_targets: Tensor | None,
        likelihood: GaussianLikelihood,
        mean_module: Mean,
        covar_module: Kernel,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor, **kwargs) -> MultivariateNormal:
        pass


def _add_cache_hook(tsr: Tensor, pred_strat: DefaultPredictionStrategy) -> Tensor:
    pass


class _VariationalStrategy(Module, ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    has_fantasy_strategy = False

    def __init__(
        self,
        model: ApproximateGP | _VariationalStrategy,
        inducing_points: Tensor,
        variational_distribution: _VariationalDistribution,
        learn_inducing_locations: bool = True,
        jitter_val: float | None = None,
    ):
        super().__init__()

        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    def _clear_cache(self) -> None:
        pass

    def _expand_inputs(self, x: Tensor, inducing_points: Tensor) -> tuple[Tensor, Tensor]:
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        pass

    @property
    def jitter_val(self) -> float:
        pass

    @jitter_val.setter
    def jitter_val(self, jitter_val: float):
        pass

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
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
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param variational_inducing_covar: If
            the distribution :math:`q(\mathbf u)` is
            Gaussian, then this variable is the covariance matrix of that Gaussian.
            Otherwise, it will be None.
        :param diag: If true and this module is in train mode, this method is allowed to skip the off-diagonal entries
            in the predictive covariance and only compute the predictive variance, whenever it's deemed more efficient
            by the underlying implementation. In that case, the off-diagonal entries in the covariance matrix of the
            returned :class:`~gpytorch.distributions.MultivariateNormal` could be arbitrary dummy values. If this
            argument is false, then this method computes the full covariance matrix even in train mode. This argument
            is ignored if this module is in eval mode, in which case the full covariance matrix is always computed.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    @cached(name="amortized_exact_gp")
    def amortized_exact_gp(self, mean_module: Module | None = None, covar_module: Module | None = None) -> ExactGP:
        pass

    def pseudo_points(self) -> tuple[Tensor, Tensor]:
        raise NotImplementedError("Each variational strategy must implement its own pseudo points method")

    def get_fantasy_model(
        self,
        inputs: Tensor,
        targets: Tensor,
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        **kwargs,
    ) -> ExactGP:
        r"""
        Performs the online variational conditioning (OVC) strategy of Maddox et al, '21 to return
        an exact GP model that incorporates the inputs and targets alongside the variational model's inducing
        points and targets.

        Currently, instead of directly updating the variational parameters (and inducing points), we instead
        return an ExactGP model rather than an updated variational GP model. This is done primarily for
        numerical stability.

        Unlike the ExactGP's call for get_fantasy_model, we enable options for mean_module and covar_module
        that allow specification of the mean / covariance. We expect that either the mean and covariance
        modules are attributes of the model itself called mean_module and covar_module respectively OR that you
        pass them into this method explicitly.

        :param inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :param mean_module: torch module describing the mean function of the GP model. Optional if
            `mean_module` is already an attribute of the variational GP.
        :param covar_module: torch module describing the covariance function of the GP model. Optional
            if `covar_module` is already an attribute of the variational GP.
        :return: An `ExactGP` model with `k + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated. We assume that there are `k` inducing points in this variational
            GP. Note that we return an `ExactGP` rather than a variational GP.

        Reference: "Conditioning Sparse Variational Gaussian Processes for Online Decision-Making,"
            Maddox, Stanton, Wilson, NeurIPS, '21
            https://papers.nips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html
        """
        pass

    def __call__(self, x: Tensor, prior: bool = False, diag: bool = True, **kwargs) -> MultivariateNormal:
        # If we're in prior mode, then we're done!
        if prior:
            if isinstance(self.model, _VariationalStrategy):
                # If the model is itself a variational strategy, we need to force it to compute the full covariance in
                # case that the model is in train mode.
                return self.model.forward(x, diag=False, **kwargs)
            else:
                # Otherwise, the model is `ApproximateGP`. So we can just call forward.
                return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                diag=diag,
                **kwargs,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=None,
                diag=diag,
                **kwargs,
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribution ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )
