#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor
from torch.distributions import Bernoulli

from ..distributions import base_distributions, MultivariateNormal
from ..functions import log_normal_cdf
from .likelihood import _OneDimensionalLikelihood


class BernoulliLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Bernoulli likelihood used for GP classification, using
    Probit regression (i.e., the latent function is warped to be in [0,1]
    using the standard Normal CDF :math:`\Phi(x)`). Given the identity
    :math:`\Phi(-x) = 1-\Phi(x)`, we can write the likelihood compactly as:

    .. math::
        \begin{equation*}
            p(Y=y|f)=\Phi((2y - 1)f)
        \end{equation*}

    .. note::
        BernoulliLikelihood has an analytic marginal distribution.

    .. note::
        The labels should take values in {0, 1}.
    """

    has_analytic_marginal: bool = True

    def __init__(self) -> None:
        return super().__init__()

    def forward(self, function_samples: Tensor, *args: Any, **kwargs: Any) -> Bernoulli:
        pass

    def log_marginal(
        self, observations: Tensor, function_dist: MultivariateNormal, *args: Any, **kwargs: Any
    ) -> Tensor:
        pass

    def marginal(self, function_dist: MultivariateNormal, *args: Any, **kwargs: Any) -> Bernoulli:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        pass

    def expected_log_prob(
        self, observations: Tensor, function_dist: MultivariateNormal, *params: Any, **kwargs: Any
    ) -> Tensor:
        pass
