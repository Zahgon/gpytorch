#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator
from torch import Tensor
from torch.autograd.function import FunctionCtx

from ..distributions import Distribution, MultivariateNormal
from .natural_variational_distribution import (
    _NaturalToMuVarSqrt,
    _NaturalVariationalDistribution,
    _phi_for_cholesky_,
    _triangular_inverse,
)


class TrilNaturalVariationalDistribution(_NaturalVariationalDistribution):
    r"""A multivariate normal :obj:`~gpytorch.variational._VariationalDistribution`,
    parameterized by the natural vector, and a triangular decomposition of the
    natural matrix (which is not the Cholesky).

    .. note::
       The :obj:`~gpytorch.variational.TrilNaturalVariationalDistribution` should only
       be used with :obj:`gpytorch.optim.NGD`, or other optimizers
       that follow exactly the gradient direction.

    .. seealso::
        The `natural gradient descent tutorial
        <examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.ipynb>`_
        for use instructions.

        The :obj:`~gpytorch.variational.NaturalVariationalDistribution`, which
        needs less iterations to make variational regression converge, at the
        cost of introducing numerical instability.

    .. note::
        The relationship of the parameter :math:`\mathbf \Theta_\text{tril_mat}`
        to the natural parameter :math:`\mathbf \Theta_\text{mat}` from
        :obj:`~gpytorch.variational.NaturalVariationalDistribution` is
        :math:`\mathbf \Theta_\text{mat} = -1/2 {\mathbf \Theta_\text{tril_mat}}^T {\mathbf \Theta_\text{tril_mat}}`.
        Note that this is not the form of the Cholesky decomposition of :math:`\boldsymbol \Theta_\text{mat}`.

    :param int num_inducing_points: Size of the variational distribution. This implies that the variational mean
        should be this size, and the variational covariance matrix should have this many rows and columns.
    :param batch_shape: Specifies an optional batch size
        for the variational parameters. This is useful for example when doing additive variational inference.
    :type batch_shape: :obj:`torch.Size`, optional
    :param float mean_init_std: (Default: 1e-3) Standard deviation of gaussian noise to add to the mean initialization.
    """

    def __init__(self, num_inducing_points: int, batch_shape: torch.Size = torch.Size([]), mean_init_std: float = 1e-3):
        super().__init__(num_inducing_points=num_inducing_points, batch_shape=batch_shape, mean_init_std=mean_init_std)
        scaled_mean_init = torch.zeros(num_inducing_points)
        neg_prec_init = torch.eye(num_inducing_points, num_inducing_points)
        scaled_mean_init = scaled_mean_init.repeat(*batch_shape, 1)
        neg_prec_init = neg_prec_init.repeat(*batch_shape, 1, 1)

        # eta1 and tril_dec(eta2) parameterization of the variational distribution
        self.register_parameter(name="natural_vec", parameter=torch.nn.Parameter(scaled_mean_init))
        self.register_parameter(name="natural_tril_mat", parameter=torch.nn.Parameter(neg_prec_init))

    def forward(self) -> Distribution:
        pass

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        pass


class _TrilNaturalToMuVarSqrt(torch.autograd.Function):
    @staticmethod
    def _forward(nat_mean: Tensor, tril_nat_covar: Tensor) -> tuple[Tensor, Tensor]:
        pass
        # return nat_mean, L

    @staticmethod
    def forward(ctx: FunctionCtx, nat_mean: Tensor, tril_nat_covar: Tensor) -> tuple[Tensor, Tensor]:
        pass

    @staticmethod
    def backward(ctx: FunctionCtx, dout_dmu: Tensor, dout_dL: Tensor) -> tuple[Tensor, Tensor]:
        pass
