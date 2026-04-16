#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from linear_operator.operators import KroneckerProductLinearOperator

from gpytorch.kernels.matern_kernel import MaternKernel

sqrt5 = math.sqrt(5)
five_thirds = 5.0 / 3.0


class Matern52KernelGrad(MaternKernel):
    r"""
    Computes a covariance matrix of the Matern52 kernel that models the covariance
    between the values and partial derivatives for inputs :math:`\mathbf{x_1}`
    and :math:`\mathbf{x_2}`.

    See :class:`gpytorch.kernels.Kernel` for descriptions of the lengthscale options.

    .. note::

        This kernel does not have an `outputscale` parameter. To add a scaling parameter,
        decorate this kernel with a :class:`gpytorch.kernels.ScaleKernel`.

    .. note::

        A perfect shuffle permutation is applied after the calculation of the matrix blocks
        in order to match the MutiTask ordering.

    The Matern52 kernel is defined as

    .. math::

        k(r) = (1 + \sqrt{5}r + \frac{5}{3} r^2) \exp(- \sqrt{5}r)

    where :math:`r` is defined as

    .. math::

        r(\mathbf{x}^m , \mathbf{x}^n) = \sqrt{\sum_d{\frac{(x^m_d - x^n_d)^2}{l^2_d}}}

    The first gradient block containing :math:`\frac{\partial k}{\partial x^n_i}` is defined as

    .. math::

        \frac{\partial k}{\partial x^n_i} = \frac{5}{3} \left( 1 + \sqrt{5}r \right) \exp(- \sqrt{5}r) \left(\frac{x^m_i - x^n_i}{l^2_i} \right)

    The second gradient block containing :math:`\frac{\partial k}{\partial x^m_j}` is defined as

    .. math::

        \frac{\partial k}{\partial x^m_j} = - \frac{5}{3} \left( 1 + \sqrt{5}r \right) \exp(- \sqrt{5}r) \left(\frac{x^m_j - x^n_j}{l^2_j} \right)

    The Hessian block containing :math:`\frac{\partial^2 k}{\partial x^m_j \partial x^n_i}` is defined as

    .. math::

        \frac{\partial^2 k}{\partial x^m_j \partial x^n_i} = - \frac{5}{3} \exp(- \sqrt{5}r) \left[ 5\left(\frac{x^m_i - x^n_i}{l^2_i} \right) \left( \frac{x^m_j - x^n_j}{l^2_j} \right) - \frac{\delta_{ij}}{l^2_i} \left( 1 + \sqrt{5}r \right) \right]

    The derivations can be found `here <https://github.com/cornellius-gp/gpytorch/pull/2512>`__.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `d` if x1 is a `n x d` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: `Positive`.)

    :ivar torch.Tensor lengthscale: The lengthscale parameter. Size/shape of parameter depends on the
        ard_num_dims and batch_shape arguments.

    Example:
        >>> x = torch.randn(10, 5)
        >>> # Non-batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.Matern52KernelGrad())
        >>> covar = covar_module(x)  # Output: LinearOperator of size (60 x 60), where 60 = n * (d + 1)
        >>>
        >>> batch_x = torch.randn(2, 10, 5)
        >>> # Batch: Simple option
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.Matern52KernelGrad())
        >>> # Batch: different lengthscale for each batch
        >>> covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.Matern52KernelGrad(batch_shape=torch.Size([2]))) # noqa: E501
        >>> covar = covar_module(x)  # Output: LinearOperator of size (2 x 60 x 60)
    """

    def __init__(self, **kwargs):

        # remove nu in case it was set
        kwargs.pop("nu", None)
        super().__init__(nu=2.5, **kwargs)

    def forward(self, x1, x2, diag=False, **params):

        pass

    def num_outputs_per_input(self, x1, x2):
        pass
