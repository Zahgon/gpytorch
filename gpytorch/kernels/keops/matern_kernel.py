#!/usr/bin/env python3

from __future__ import annotations

import math

from linear_operator.operators import KernelLinearOperator
from torch import Tensor

from .keops_kernel import _Anysor, _lazify_and_expand_inputs, KeOpsKernel


def _covar_func(x1: _Anysor, x2: _Anysor, nu: float = 2.5, **params) -> _Anysor:
    pass


class MaternKernel(KeOpsKernel):
    """
    Implements the Matern kernel using KeOps as a driver for kernel matrix multiplies.

    This class can be used as a drop in replacement for :class:`gpytorch.kernels.MaternKernel` in most cases,
    and supports the same arguments.

    :param nu: (Default: 2.5) The smoothness parameter.
    :type nu: float (0.5, 1.5, or 2.5)
    :param ard_num_dims: (Default: `None`) Set this if you want a separate lengthscale for each
        input dimension. It should be `d` if x1 is a `... x n x d` matrix.
    :type ard_num_dims: int, optional
    :param batch_shape: (Default: `None`) Set this if you want a separate lengthscale for each
         batch of input data. It should be `torch.Size([b1, b2])` for a `b1 x b2 x n x m` kernel output.
    :type batch_shape: torch.Size, optional
    :param active_dims: (Default: `None`) Set this if you want to
        compute the covariance of only a few input dimensions. The ints
        corresponds to the indices of the dimensions.
    :type active_dims: Tuple(int)
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional
    """

    has_lengthscale = True

    def __init__(self, nu: float = 2.5, **kwargs):
        if nu not in {0.5, 1.5, 2.5}:
            raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        super().__init__(**kwargs)
        self.nu = nu

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> KernelLinearOperator:
        pass
