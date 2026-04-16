#!/usr/bin/env python3

from __future__ import annotations

import torch
from torch import Tensor

from .. import settings
from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class CylindricalKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Cylindrical Kernel between
    inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`.
    It was proposed in `BOCK: Bayesian Optimization with Cylindrical Kernels`.
    See http://proceedings.mlr.press/v80/oh18a.html for more details

    .. note::
        The data must lie completely within the unit ball.

    Args:
        num_angular_weights (int):
            The number of components in the angular kernel
        radial_base_kernel (gpytorch.kernel):
            The base kernel for computing the radial kernel
        batch_size (int, optional):
            Set this if the data is batch of input data.
            It should be `b` if x1 is a `b x n x d` tensor. Default: `1`
        eps (float):
            Small floating point number used to improve numerical stability
            in kernel computations. Default: `1e-6`
        param_transform (function, optional):
            Set this if you want to use something other than softplus to ensure positiveness of parameters.
        inv_param_transform (function, optional):
            Set this to allow setting parameters directly in transformed space and sampling from priors.
            Automatically inferred for common transformations such as torch.exp or torch.nn.functional.softplus.
    """

    def __init__(
        self,
        num_angular_weights: int,
        radial_base_kernel: Kernel,
        eps: float | None = 1e-6,
        angular_weights_prior: Prior | None = None,
        angular_weights_constraint: Interval | None = None,
        alpha_prior: Prior | None = None,
        alpha_constraint: Interval | None = None,
        beta_prior: Prior | None = None,
        beta_constraint: Interval | None = None,
        **kwargs,
    ):
        if angular_weights_constraint is None:
            angular_weights_constraint = Positive()

        if alpha_constraint is None:
            alpha_constraint = Positive()

        if beta_constraint is None:
            beta_constraint = Positive()

        super().__init__(**kwargs)
        self.num_angular_weights = num_angular_weights
        self.radial_base_kernel = radial_base_kernel
        self.eps = eps

        self.register_parameter(
            name="raw_angular_weights",
            parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, num_angular_weights)),
        )
        self.register_constraint("raw_angular_weights", angular_weights_constraint)
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_alpha", alpha_constraint)
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        self.register_constraint("raw_beta", beta_constraint)

        if angular_weights_prior is not None:
            if not isinstance(angular_weights_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(angular_weights_prior).__name__)
            self.register_prior(
                "angular_weights_prior",
                angular_weights_prior,
                lambda m: m.angular_weights,
                lambda m, v: m._set_angular_weights(v),
            )
        if alpha_prior is not None:
            if not isinstance(alpha_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(alpha_prior).__name__)
            self.register_prior("alpha_prior", alpha_prior, lambda m: m.alpha, lambda m, v: m._set_alpha(v))
        if beta_prior is not None:
            if not isinstance(beta_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(beta_prior).__name__)
            self.register_prior("beta_prior", beta_prior, lambda m: m.beta, lambda m, v: m._set_beta(v))

    @property
    def angular_weights(self) -> Tensor:
        pass

    @angular_weights.setter
    def angular_weights(self, value: Tensor) -> None:
        pass

    @property
    def alpha(self) -> Tensor:
        pass

    @alpha.setter
    def alpha(self, value: Tensor) -> None:
        pass

    def _set_alpha(self, value: Tensor | float) -> None:
        # Used by the alpha_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self) -> Tensor:
        pass

    @beta.setter
    def beta(self, value: Tensor) -> None:
        pass

    def _set_beta(self, value: Tensor | float) -> None:
        # Used by the beta_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def forward(self, x1: Tensor, x2: Tensor, diag: bool | None = False, **params) -> Tensor:

        pass

    def kuma(self, x: Tensor) -> Tensor:
        pass

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        pass
