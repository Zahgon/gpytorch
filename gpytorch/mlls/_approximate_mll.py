#!/usr/bin/env python3

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from .marginal_log_likelihood import MarginalLogLikelihood


class _ApproximateMarginalLogLikelihood(MarginalLogLikelihood, ABC):
    r"""
    An approximate marginal log likelihood (typically a bound) for approximate GP models.
    We expect that model is a :obj:`gpytorch.models.ApproximateGP`.

    Args:
        likelihood (:obj:`gpytorch.likelihoods.Likelihood`):
            The likelihood for the model
        model (:obj:`gpytorch.models.ApproximateGP`):
            The approximate GP model
        num_data (int):
            The total number of training data points (necessary for SGD)
        beta (float - default 1.):
            A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference
            (as derived in `Scalable Variational Gaussian Process Classification`_).
            Setting it to anything less than 1 reduces the regularization effect of the model
            (similarly to what was proposed in `the beta-VAE paper`_).
        combine_terms (bool):
            Whether or not to sum the expected NLL with the KL terms (default True)
    """

    def __init__(self, likelihood, model, num_data, beta=1.0, combine_terms=True):
        super().__init__(likelihood, model)
        self.combine_terms = combine_terms
        self.num_data = num_data
        self.beta = beta

    @abstractmethod
    def _log_likelihood_term(self, approximate_dist_f, target, **kwargs):
        raise NotImplementedError

    def forward(self, approximate_dist_f, target, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            approximate_dist_f (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            target (`torch.Tensor`):
                :math:`\mathbf y` The target values

        Keyword Args:
            Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        pass
