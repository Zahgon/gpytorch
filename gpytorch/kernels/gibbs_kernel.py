#!/usr/bin/env python3

from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .kernel import Kernel


class GibbsKernel(Kernel):
    r"""
    Gibbs kernel with input-dependent lengthscale :math:`\ell(x)` (Gibbs, 1997)

    .. math::
        k(x, x') = \sqrt{\frac{2\ell(x)\ell(x')}{\ell(x)^2 + \ell(x')^2}}
        \exp\left(-\frac{(x-x')^2}{\ell(x)^2 + \ell(x')^2}\right)

    :param lengthscale_fn: A callable torch.nn.Module mapping inputs to
        positive lengthscales. Must output tensors of shape (... x N x 1)
        for input of shape (... x N x D)
    :type lengthscale_fn: torch.nn.Module

    Example::

        class LengthscaleMLP(torch.nn.Module):
            def __init__(self, in_dim=1, hidden=32):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, 1),
                    torch.nn.Softplus(),
                )

            def forward(self, x):
                return self.net(x)

        kernel = GibbsKernel(lengthscale_fn=LengthscaleMLP(in_dim=1))
    """

    is_stationary = False
    has_lengthscale = False

    def __init__(self, lengthscale_fn: nn.Module, **kwargs):
        if kwargs.get("ard_num_dims") is not None:
            raise NotImplementedError("GibbsKernel does not support ARD.")
        super().__init__(**kwargs)
        self.lengthscale_fn = lengthscale_fn

    # Update batch_shape explicitly:
    # Base class derives new batch_shape from parameters,
    # but GibbsKernel has none
    def __getitem__(self, index):
        if len(self.batch_shape) == 0:
            return self
        new_kernel = deepcopy(self)
        index = index if isinstance(index, tuple) else (index,)
        new_kernel.batch_shape = torch.empty(self.batch_shape)[index].shape
        return new_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        pass
