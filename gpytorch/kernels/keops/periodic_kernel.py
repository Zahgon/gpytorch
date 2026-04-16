#!/usr/bin/env python3

from __future__ import annotations

import math

from linear_operator.operators import KernelLinearOperator
from torch import Tensor

from ..periodic_kernel import PeriodicKernel as GPeriodicKernel
from .keops_kernel import _Anysor, _lazify_and_expand_inputs, KeOpsKernel


def _covar_func(x1: _Anysor, x2: _Anysor, lengthscale: Tensor, **kwargs) -> _Anysor:
    pass


# subclass from original periodic kernel to reduce code duplication
class PeriodicKernel(KeOpsKernel, GPeriodicKernel):
    """
    Implements the Periodic Kernel using KeOps as a driver for kernel matrix multiplies.

    This class can be used as a drop in replacement for :class:`gpytorch.kernels.PeriodicKernel` in most cases,
    and supports the same arguments.

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
    :param period_length_prior: (Default: `None`)
        Set this if you want to apply a prior to the period length parameter.
    :type period_length_prior: ~gpytorch.priors.Prior, optional
    :param period_length_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the period length parameter.
    :type period_length_constraint: ~gpytorch.constraints.Interval, optional
    :param lengthscale_prior: (Default: `None`)
        Set this if you want to apply a prior to the lengthscale parameter.
    :type lengthscale_prior: ~gpytorch.priors.Prior, optional
    :param lengthscale_constraint: (Default: `Positive`) Set this if you want
        to apply a constraint to the lengthscale parameter.
    :type lengthscale_constraint: ~gpytorch.constraints.Interval, optional

    :var torch.Tensor period_length: The period length parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.
    """

    has_lengthscale = True

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **kwargs) -> KernelLinearOperator:
        pass
