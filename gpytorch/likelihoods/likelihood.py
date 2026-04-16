#!/usr/bin/env python3

from __future__ import annotations

import math
import warnings
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Distribution as _Distribution

from .. import settings
from ..distributions import base_distributions, MultivariateNormal
from ..module import Module
from ..utils.quadrature import GaussHermiteQuadrature1D
from ..utils.warnings import GPInputWarning


class _Likelihood(Module, ABC):
    has_analytic_marginal: bool = False

    def __init__(self, max_plate_nesting: int = 1) -> None:
        super().__init__()
        self.max_plate_nesting: int = max_plate_nesting

    def _draw_likelihood_samples(
        self, function_dist: MultivariateNormal, *args: Any, sample_shape: torch.Size | None = None, **kwargs: Any
    ) -> _Distribution:
        pass

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        pass

    @abstractmethod
    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> _Distribution:
        raise NotImplementedError

    def get_fantasy_likelihood(self, **kwargs: Any) -> _Likelihood:
        pass

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        pass

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> _Distribution:
        pass

    def __call__(self, input: Tensor | MultivariateNormal, *args: Any, **kwargs: Any) -> _Distribution:
        # Conditional
        if torch.is_tensor(input):
            return super().__call__(input, *args, **kwargs)  # pyre-ignore[7]
        # Marginal
        elif isinstance(input, MultivariateNormal):
            return self.marginal(input, *args, **kwargs)
        # Error
        else:
            raise RuntimeError(
                "Likelihoods expects a MultivariateNormal input to make marginal predictions, or a "
                "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
            )


try:
    import pyro

    class Likelihood(_Likelihood):
        r"""
        A Likelihood in GPyTorch specifies the mapping from latent function values
        :math:`f(\mathbf X)` to observed labels :math:`y`.

        For example, in the case of regression this might be a Gaussian
        distribution, as :math:`y(\mathbf x)` is equal to :math:`f(\mathbf x)` plus Gaussian noise:

        .. math::
            y(\mathbf x) = f(\mathbf x) + \epsilon, \:\:\:\: \epsilon \sim N(0,\sigma^{2}_{n} \mathbf I)

        In the case of classification, this might be a Bernoulli distribution,
        where the probability that :math:`y=1` is given by the latent function
        passed through some sigmoid or probit function:

        .. math::
            y(\mathbf x) = \begin{cases}
                1 & \text{w/ probability} \:\: \sigma(f(\mathbf x)) \\
                0 & \text{w/ probability} \:\: 1-\sigma(f(\mathbf x))
            \end{cases}

        In either case, to implement a likelihood function, GPyTorch only
        requires a forward method that computes the conditional distribution
        :math:`p(y \mid f(\mathbf x))`.

        :param bool has_analytic_marginal: Whether or not the marginal distribution :math:`p(\mathbf y)`
            can be computed in closed form. (See :meth:`~gpytorch.likelihoods.Likelihood.__call__` docstring.)
        :param max_plate_nesting: (For Pyro integration only.) How many batch dimensions are in the function.
            This should be modified if the likelihood uses plated random variables. (Default = 1)
            This should be modified if the likelihood uses plated random variables. (Default = 1)
        :param str name_prefix: (For Pyro integration only.) Prefix to assign to named Pyro latent variables.
        :param int num_data: (For Pyro integration only.) Total amount of observations.
        """

        @property
        def num_data(self) -> int:
            pass

        @num_data.setter
        def num_data(self, val: int) -> None:
            pass

        @property
        def name_prefix(self) -> str:
            pass

        @name_prefix.setter
        def name_prefix(self, val: str) -> None:
            pass

        def _draw_likelihood_samples(
            self, function_dist: _Distribution, *args: Any, sample_shape: torch.Size | None = None, **kwargs: Any
        ) -> _Distribution:
            pass

        def expected_log_prob(
            self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
        ) -> Tensor:
            r"""
            (Used by :obj:`~gpytorch.mlls.VariationalELBO` for variational inference.)

            Computes the expected log likelihood, where the expectation is over the GP variational distribution.

            .. math::
                \sum_{\mathbf x, y} \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ \log p \left( y \mid f(\mathbf x) \right) \right]

            :param observations: Values of :math:`y`.
            :param function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            """
            pass

        @abstractmethod
        def forward(
            self, function_samples: Tensor, *args: Any, data: dict[str, Tensor] = {}, **kwargs: Any
        ) -> _Distribution:
            r"""
            Computes the conditional distribution :math:`p(\mathbf y \mid
            \mathbf f, \ldots)` that defines the likelihood.

            :param function_samples: Samples from the function (:math:`\mathbf f`)
            :param data: (Pyro integration only.) Additional variables that the likelihood needs to condition
                on. The keys of the dictionary will correspond to Pyro sample sites
                in the likelihood's model/guide.
            :param args: Additional args
            :param kwargs: Additional kwargs
            """
            raise NotImplementedError

        def get_fantasy_likelihood(self, **kwargs: Any) -> _Likelihood:
            """"""
            pass

        def log_marginal(
            self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
        ) -> Tensor:
            r"""
            (Used by :obj:`~gpytorch.mlls.PredictiveLogLikelihood` for approximate inference.)

            Computes the log marginal likelihood of the approximate predictive distribution

            .. math::
                \sum_{\mathbf x, y} \log \mathbb{E}_{q\left( f(\mathbf x) \right)}
                \left[ p \left( y \mid f(\mathbf x) \right) \right]

            Note that this differs from :meth:`expected_log_prob` because the :math:`log` is on the outside
            of the expectation.

            :param observations: Values of :math:`y`.
            :param function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            """
            pass

        def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> _Distribution:
            r"""
            Computes a predictive distribution :math:`p(y^* | \mathbf x^*)` given either a posterior
            distribution :math:`p(\mathbf f | \mathcal D, \mathbf x)` or a
            prior distribution :math:`p(\mathbf f|\mathbf x)` as input.

            With both exact inference and variational inference, the form of
            :math:`p(\mathbf f|\mathcal D, \mathbf x)` or :math:`p(\mathbf f|
            \mathbf x)` should usually be Gaussian. As a result, function_dist
            should usually be a :obj:`~gpytorch.distributions.MultivariateNormal` specified by the mean and
            (co)variance of :math:`p(\mathbf f|...)`.

            :param function_dist: Distribution for :math:`f(x)`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            :return: The marginal distribution, or samples from it.
            """
            pass

        def pyro_guide(self, function_dist: MultivariateNormal, target: Tensor, *args: Any, **kwargs: Any) -> None:
            r"""
            (For Pyro integration only).

            Part of the guide function for the likelihood.
            This should be re-defined if the likelihood contains any latent variables that need to be infered.

            :param function_dist: Distribution of latent function
                :math:`q(\mathbf f)`.
            :param target: Observed :math:`\mathbf y`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            """
            pass

        def pyro_model(self, function_dist: MultivariateNormal, target: Tensor, *args: Any, **kwargs: Any) -> Tensor:
            r"""
            (For Pyro integration only).

            Part of the model function for the likelihood.
            It should return the
            This should be re-defined if the likelihood contains any latent variables that need to be infered.

            :param function_dist: Distribution of latent function
                :math:`p(\mathbf f)`.
            :param target: Observed :math:`\mathbf y`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            """
            pass

        def sample_target(self, output_dist: MultivariateNormal, target: Tensor) -> Tensor:
            pass

        def __call__(self, input: Tensor | MultivariateNormal, *args: Any, **kwargs: Any) -> _Distribution:
            r"""
            Calling this object does one of two things:

            1. If likelihood is called with a :class:`torch.Tensor` object, then it is
               assumed that the input is samples from :math:`f(\mathbf x)`. This
               returns the *conditional* distribution :math:`p(y|f(\mathbf x))`.

            .. code-block:: python

                f = torch.randn(20)
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                conditional = likelihood(f)
                print(type(conditional), conditional.batch_shape, conditional.event_shape)
                # >>> torch.distributions.Normal, torch.Size([20]), torch.Size([])

            2. If likelihood is called with a :class:`~gpytorch.distribution.MultivariateNormal` object,
               then it is assumed that the input is the distribution :math:`f(\mathbf x)`.
               This returns the *marginal* distribution :math:`p(y|\mathbf x)`.

               The form of the marginal distribution depends on the likelihood.
               For :class:`~gpytorch.likelihoods.BernoulliLikelihood` and
               :class:`~gpytorch.likelihoods.GaussianLikelihood` objects, the marginal distribution
               can be computed analytically, and the likelihood returns the analytic distribution.
               For most other likelihoods, there is no analytic form for the marginal,
               and so the likelihood instead returns a batch of Monte Carlo samples from the marginal.

            .. code-block:: python

                mean = torch.randn(20)
                covar = linear_operator.operators.DiagLinearOperator(torch.ones(20))
                f = gpytorch.distributions.MultivariateNormal(mean, covar)

                # Analytic marginal computation - Bernoulli and Gaussian likelihoods only
                analytic_marginal_likelihood = gpytorch.likelihoods.GaussianLikelihood()
                marginal = analytic_marginal_likelihood(f)
                print(type(marginal), marginal.batch_shape, marginal.event_shape)
                # >>> <class 'gpytorch.distributions.multivariate_normal.MultivariateNormal'> torch.Size([]) torch.Size([20])  # noqa: E501

                # MC marginal computation - all other likelihoods
                mc_marginal_likelihood = gpytorch.likelihoods.BetaLikelihood()
                with gpytorch.settings.num_likelihood_samples(15):
                    marginal = mc_marginal_likelihood(f)
                print(type(marginal), marginal.batch_shape, marginal.event_shape)
                # >>> <class 'torch.distributions.beta.Beta'> torch.Size([15, 20]) torch.Size([])
                # The batch_shape torch.Size([15, 20]) represents 15 MC samples for 20 data points.

            .. note::

                If a Likelihood supports analytic marginals, the :attr:`has_analytic_marginal` property will be True.
                If a Likelihood does not support analytic marginals, you can set the number of Monte Carlo
                samples using the :class:`gpytorch.settings.num_likelihood_samples` context manager.

            :param input: Either a (... x N) sample from :math:`\mathbf f`
                or a (... x N) MVN distribution of :math:`\mathbf f`.
            :param args: Additional args (passed to the forward function).
            :param kwargs: Additional kwargs (passed to the forward function).
            :return: Either a conditional :math:`p(\mathbf y \mid \mathbf f)`
                or marginal :math:`p(\mathbf y)`
                based on whether :attr:`input` is a Tensor or a MultivariateNormal (see above).
            """
            # Conditional
            if torch.is_tensor(input):
                return super().__call__(input, *args, **kwargs)
            # Marginal
            elif any(
                [
                    isinstance(input, MultivariateNormal),
                    isinstance(input, pyro.distributions.Normal),  # pyre-ignore[16]
                    (
                        isinstance(input, pyro.distributions.Independent)  # pyre-ignore[16]
                        and isinstance(input.base_dist, pyro.distributions.Normal)  # pyre-ignore[16]
                    ),
                ]
            ):
                return self.marginal(input, *args, **kwargs)  # pyre-ignore[6]
            # Error
            else:
                raise RuntimeError(
                    "Likelihoods expects a MultivariateNormal or Normal input to make marginal predictions, or a "
                    "torch.Tensor for conditional predictions. Got a {}".format(input.__class__.__name__)
                )

except ImportError:

    class Likelihood(_Likelihood):
        @property
        def num_data(self) -> int:
            pass

        @num_data.setter
        def num_data(self, val: int) -> None:
            pass

        @property
        def name_prefix(self) -> str:
            pass

        @name_prefix.setter
        def name_prefix(self, val: str) -> None:
            pass


class _OneDimensionalLikelihood(Likelihood, ABC):
    r"""
    A specific case of :obj:`~gpytorch.likelihoods.Likelihood` when the GP represents a one-dimensional
    output. (I.e. for a specific :math:`\mathbf x`, :math:`f(\mathbf x) \in \mathbb{R}`.)

    Inheriting from this likelihood reduces the variance when computing approximate GP objective functions
    by using 1D Gauss-Hermite quadrature.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.quadrature = GaussHermiteQuadrature1D()

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        pass

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        pass
