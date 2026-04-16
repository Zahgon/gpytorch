#!/usr/bin/env python3

from __future__ import annotations

from torch.distributions import kl_divergence

from ..distributions import MultivariateNormal
from .added_loss_term import AddedLossTerm


class KLGaussianAddedLossTerm(AddedLossTerm):
    r"""
    This class is used by variational GPLVM models.
    It adds the KL divergence between two multivariate Gaussian distributions:
    scaled by the size of the data and the number of output dimensions.

    .. math::

        D_\text{KL} \left( q(\mathbf x) \Vert p(\mathbf x) \right)


    :param q_x: The MVN distribution :math:`q(\mathbf x)`.
    :param p_x: The MVN distribution :math:`p(\mathbf x)`.
    :param n: Size of the latent space.
    :param data_dim: Dimensionality of the :math:`\mathbf Y` values.
    """

    def __init__(self, q_x: MultivariateNormal, p_x: MultivariateNormal, n: int, data_dim: int):
        super().__init__()
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.data_dim = data_dim

    def loss(self):
        pass
