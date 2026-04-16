from __future__ import annotations

import torch

from linear_operator.operators import DiagLinearOperator, LinearOperator, MatmulLinearOperator
from torch import Tensor

from gpytorch.variational.variational_strategy import VariationalStrategy


class QuadFormDiagonal(torch.autograd.Function):
    r"""A custom autograd function computing the diagonal of a quadratic form.

    This function computes `torch.diag(B' A B)` where `A` is a symmetric matrix. The backward pass saves a large matmul
    compared to PyTorch's default autograd engine when `B` has way more columns than rows.
    """

    @staticmethod
    def forward(ctx, matrix: Tensor, rhs: Tensor):
        r"""The forward pass computing the diagonal of a quadratic form. Note that it does not form `B' A B` explicitly.

        :param matrix: A symmetric matrix of size `(..., M, M)`.
        :param rhs: The right-hand side vector of size `(..., M, N)`.

        :return: The quadratic form diagonal of size `(..., N)`.
        """
        pass

    @staticmethod
    def backward(ctx, d_diag: Tensor):
        pass


class LargeBatchVariationalStrategy(VariationalStrategy):
    r"""A fast variational strategy implementation optimized for large batch stochastic training on data center GPUs.

    This implementation has two assumptions on the use case:
    1. FP64 operations (in particular triangular solve and matmul) on data center GPUs are not much slower than FP32;
    2. The batch size is very large while the number of inducing points is moderate.

    This implementation speeds up the standard `VariationalStrategy` in two ways:
    1. Group the middle term `K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2}` when computing the predictive covariance, which saves a
    large triangular solve in the forward pass;
    2. Use a custom autograd function computing the diagonal of `K_XZ @ middle_term @ K_ZX` in train mode, which saves
    a large matmul in the backward pass.

    NOTE: Grouping the middle term is not numerically friendly, and thus we have to use double precision to stabilize
    the computation. As a result, this implementation is expected to be slow on CPUs and consumer GPUs. Those who use
    CPUs and consumer cards should use `VariationalStrategy` instead.
    """

    def _clear_cache(self) -> None:
        # Clear cached inference terms before calling parent's _clear_cache
        pass

    def _compute_predictive_updates(
        self,
        chol: LinearOperator,
        induc_data_covar: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar: LinearOperator | None,
        prior_covar: LinearOperator,
        diag: bool = True,
    ) -> tuple[Tensor, LinearOperator]:
        pass
