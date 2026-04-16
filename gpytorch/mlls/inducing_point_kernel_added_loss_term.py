#!/usr/bin/env python3

from __future__ import annotations

import torch

from ..distributions import MultivariateNormal
from ..likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from .added_loss_term import AddedLossTerm


class InducingPointKernelAddedLossTerm(AddedLossTerm):
    r"""
    An added loss term that computes the additional "regularization trace term" of the SGPR objective function.

    .. math::
        -\frac{1}{2 \sigma^2} \text{Tr} \left( \mathbf K_{\mathbf X \mathbf X} - \mathbf Q \right)


    where :math:`\mathbf Q = \mathbf K_{\mathbf X \mathbf Z} \mathbf K_{\mathbf Z \mathbf Z}^{-1}
    \mathbf K_{\mathbf Z \mathbf X}` is the Nystrom approximation of :math:`\mathbf K_{\mathbf X \mathbf X}`
    given by inducing points :math:`\mathbf Z`, and :math:`\sigma^2` is the observational noise
    of the Gaussian likelihood.

    See `Titsias, 2009`_, Eq. 9 for more more information.

    :param prior_dist: A multivariate normal :math:`\mathcal N ( \mathbf 0, \mathbf K_{\mathbf X \mathbf X} )`
        with covariance matrix :math:`\mathbf K_{\mathbf X \mathbf X}`.
    :param variational_dist: A multivariate normal :math:`\mathcal N ( \mathbf 0, \mathbf Q`
        with covariance matrix :math:`\mathbf Q = \mathbf K_{\mathbf X \mathbf Z}
        \mathbf K_{\mathbf Z \mathbf Z}^{-1} \mathbf K_{\mathbf Z \mathbf X}`.
    :param likelihood: The Gaussian likelihood with observational noise :math:`\sigma^2`.

    .. _Titsias, 2009:
        https://arxiv.org/pdf/1302.4245.pdf
    """

    def __init__(
        self, prior_dist: MultivariateNormal, variational_dist: MultivariateNormal, likelihood: GaussianLikelihood
    ):
        self.prior_dist = prior_dist
        self.variational_dist = variational_dist
        self.likelihood = likelihood

    def loss(self, *params) -> torch.Tensor:
        pass
