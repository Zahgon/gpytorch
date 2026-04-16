#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import DiagLinearOperator, LinearOperator, MatmulLinearOperator, SumLinearOperator
from linear_operator.utils import linear_cg
from torch import Tensor
from torch.autograd.function import FunctionCtx

from .. import settings
from ..distributions import Delta, Distribution, MultivariateNormal
from ..module import Module
from ..utils.memoize import cached
from ._variational_strategy import _VariationalStrategy
from .natural_variational_distribution import NaturalVariationalDistribution


class _NgdInterpTerms(torch.autograd.Function):
    """
    This function takes in

        - the kernel interpolation term K_ZZ^{-1/2} k_ZX
        - the natural parameters of the variational distribution

    and returns

        - the predictive distribution mean/covariance
        - the inducing KL divergence KL( q(u) || p(u))

    However, the gradients will be with respect to the **cannonical parameters**
    of the variational distribution, rather than the **natural parameters**.
    This corresponds to performing natural gradient descent on the variational distribution.
    """

    @staticmethod
    def forward(
        ctx: FunctionCtx,
        interp_term: torch.Tensor,
        natural_vec: torch.Tensor,
        natural_mat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute precision
        pass

    @staticmethod
    def backward(
        ctx: FunctionCtx, interp_mean_grad: torch.Tensor, interp_var_grad: torch.Tensor, kl_div_grad: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        # Get the saved terms
        pass


class CiqVariationalStrategy(_VariationalStrategy):
    r"""
    Similar to :class:`~gpytorch.variational.VariationalStrategy`,
    except the whitening operation is performed using Contour Integral Quadrature
    rather than Cholesky (see `Pleiss et al. (2020)`_ for more info).
    See the `CIQ-SVGP tutorial`_ for an example.

    Contour Integral Quadrature uses iterative matrix-vector multiplication to approximate
    the :math:`\mathbf K_{\mathbf Z \mathbf Z}^{-1/2}` matrix used for the whitening operation.
    This can be more efficient than the standard variational strategy for large numbers
    of inducing points (e.g. :math:`M > 1000`) or when the inducing points have structure
    (e.g. they lie on an evenly-spaced grid).

    .. note::

        It is recommended that this object is used in conjunction with
        :obj:`~gpytorch.variational.NaturalVariationalDistribution` and
        `natural gradient descent`_.

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Pleiss et al. (2020):
        https://arxiv.org/pdf/2006.11267.pdf
    .. _CIQ-SVGP tutorial:
        examples/04_Variational_and_Approximate_GPs/SVGP_CIQ.html
    .. _natural gradient descent:
        examples/04_Variational_and_Approximate_GPs/Natural_Gradient_Descent.html
    """

    def _ngd(self) -> bool:
        return isinstance(self._variational_distribution, NaturalVariationalDistribution)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        pass

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
        pass

    def forward(
        self,
        x: torch.Tensor,
        inducing_points: torch.Tensor,
        inducing_values: torch.Tensor,
        variational_inducing_covar: LinearOperator | None = None,
        diag: bool = True,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        pass

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.

        :rtype: torch.Tensor
        """
        if self._ngd():
            if hasattr(self, "_memoize_cache") and "kl" in self._memoize_cache:
                return self._memoize_cache["kl"]
            else:
                raise RuntimeError(
                    "KL divergence for NGD-CIQ should be computed during forward calls."
                    "This is probably a bug in GPyTorch."
                )
        else:
            return super().kl_divergence()

    def __call__(self, x: torch.Tensor, prior: bool = False, diag: bool = True, **kwargs) -> MultivariateNormal:
        # This is mostly the same as _VariationalStrategy.__call__()
        # but with special rules for natural gradient descent (to prevent O(M^3) computation)

        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            if self._ngd():
                noise = torch.randn_like(self.prior_distribution.mean).mul_(1e-3)
                eye = torch.eye(noise.size(-1), dtype=noise.dtype, device=noise.device).mul(-0.5)
                self._variational_distribution.natural_vec.data.copy_(noise)
                self._variational_distribution.natural_mat.data.copy_(eye)
                self.variational_params_initialized.fill_(1)
            else:
                prior_dist = self.prior_distribution
                self._variational_distribution.initialize_variational_distribution(prior_dist)
                self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get q(f)
        if self._ngd():
            return Module.__call__(
                self,
                x,
                inducing_points,
                inducing_values=None,
                variational_inducing_covar=None,
                diag=diag,
                **kwargs,
            )
        else:
            # Get p(u)/q(u)
            variational_dist_u = self.variational_distribution

            if isinstance(variational_dist_u, MultivariateNormal):
                return Module.__call__(
                    self,
                    x,
                    inducing_points,
                    inducing_values=variational_dist_u.mean,
                    variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                    diag=diag,
                    **kwargs,
                )
            elif isinstance(variational_dist_u, Delta):
                return Module.__call__(
                    self,
                    x,
                    inducing_points,
                    inducing_values=variational_dist_u.mean,
                    variational_inducing_covar=None,
                    diag=diag,
                    **kwargs,
                )
            else:
                raise RuntimeError(
                    f"Invalid variational distribution ({type(variational_dist_u)}). "
                    "Expected a multivariate normal or a delta distribution."
                )
