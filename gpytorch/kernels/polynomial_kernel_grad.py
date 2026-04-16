#!/usr/bin/env python3

from __future__ import annotations

import torch

from .polynomial_kernel import PolynomialKernel


class PolynomialKernelGrad(PolynomialKernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool | None = False,
        last_dim_is_batch: bool | None = False,
        **params,
    ) -> torch.Tensor:
        pass

    def num_outputs_per_input(self, x1, x2):
        pass
