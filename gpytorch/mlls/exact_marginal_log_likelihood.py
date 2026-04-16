#!/usr/bin/env python3

from __future__ import annotations

from linear_operator.operators import MaskedLinearOperator

from .. import settings
from ..distributions import MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from .marginal_log_likelihood import MarginalLogLikelihood


class ExactMarginalLogLikelihood(MarginalLogLikelihood):
    """
    The exact marginal log likelihood (MLL) for an exact Gaussian process with a
    Gaussian likelihood.

    .. note::
        This module will not work with anything other than a :obj:`~gpytorch.likelihoods.GaussianLikelihood`
        and a :obj:`~gpytorch.models.ExactGP`. It also cannot be used in conjunction with
        stochastic optimization.

    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood for the model
    :param ~gpytorch.models.ExactGP model: The exact GP model

    Example:
        >>> # model is a gpytorch.models.ExactGP
        >>> # likelihood is a gpytorch.likelihoods.Likelihood
        >>> mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        >>>
        >>> output = model(train_x)
        >>> loss = -mll(output, train_y)
        >>> loss.backward()
    """

    def __init__(self, likelihood, model):
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("Likelihood must be Gaussian for exact inference")
        super().__init__(likelihood, model)

    def _add_other_terms(self, res, params):
        # Add additional terms (SGPR / learned inducing points, heteroskedastic likelihood models)
        pass

    def forward(self, function_dist, target, *params, **kwargs):
        r"""
        Computes the MLL given :math:`p(\mathbf f)` and :math:`\mathbf y`.

        :param ~gpytorch.distributions.MultivariateNormal function_dist: :math:`p(\mathbf f)`
            the outputs of the latent function (the :obj:`gpytorch.models.ExactGP`)
        :param torch.Tensor target: :math:`\mathbf y` The target values
        :rtype: torch.Tensor
        :return: Exact MLL. Output shape corresponds to batch shape of the model/input data.
        """
        pass
