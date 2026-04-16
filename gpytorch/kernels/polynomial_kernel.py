#!/usr/bin/env python3

from __future__ import annotations

import torch

from ..constraints import Interval, Positive
from ..priors import Prior
from .kernel import Kernel


class PolynomialKernel(Kernel):
    r"""
    Computes a covariance matrix based on the Polynomial kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::
        \begin{equation*}
            k_\text{Poly}(\mathbf{x_1}, \mathbf{x_2}) = (\mathbf{x_1}^\top
            \mathbf{x_2} + c)^{d}.
        \end{equation*}

    where

    * :math:`c` is an offset parameter.

    Args:
        offset_prior (:class:`gpytorch.priors.Prior`):
            Prior over the offset parameter (default `None`).
        offset_constraint (Constraint, optional):
            Constraint to place on offset parameter. Default: `Positive`.
        active_dims (list):
            List of data dimensions to operate on.
            `len(active_dims)` should equal `num_dimensions`.
    """

    def __init__(
        self,
        power: int,
        offset_prior: Prior | None = None,
        offset_constraint: Interval | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if offset_constraint is None:
            offset_constraint = Positive()

        self.register_parameter(name="raw_offset", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))

        # We want the power to be a float so we dont have to worry about its device / dtype.
        if torch.is_tensor(power):
            if power.numel() > 1:
                raise RuntimeError("Cant create a Polynomial kernel with more than one power")
            else:
                power = power.item()

        self.power = power

        if offset_prior is not None:
            if not isinstance(offset_prior, Prior):
                raise TypeError("Expected gpytorch.priors.Prior but got " + type(offset_prior).__name__)
            self.register_prior("offset_prior", offset_prior, lambda m: m.offset, lambda m, v: m._set_offset(v))

        self.register_constraint("raw_offset", offset_constraint)

    @property
    def offset(self) -> torch.Tensor:
        pass

    @offset.setter
    def offset(self, value: torch.Tensor) -> None:
        pass

    def _set_offset(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool | None = False,
        last_dim_is_batch: bool | None = False,
        **params,
    ) -> torch.Tensor:
        pass
