#!/usr/bin/env python3

from __future__ import annotations

import warnings

import torch
from linear_operator import to_dense

from ..constraints import Positive
from .kernel import Kernel


class NewtonGirardAdditiveKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        num_dims: int,
        max_degree: int | None = None,
        active_dims: tuple[int, ...] | None = None,
        **kwargs,
    ):
        """Create an Additive Kernel a la https://arxiv.org/abs/1112.4394 using Newton-Girard Formulae

        :param base_kernel: a base 1-dimensional kernel. NOTE: put ard_num_dims=d in the base kernel...
        :param max_degree: the maximum numbers of kernel degrees to compute
        :param active_dims:
        :param kwargs:
        """

        warnings.warn(
            "NewtonGirardAdditiveKernel is deprecated, and will be removed in GPyTorch 2.0. "
            'Please refer to the "Kernels with Additive or Product Structure" tutorial '
            "in the GPyTorch docs for how to implement GPs with additive structure.",
            DeprecationWarning,
        )
        super().__init__(active_dims=active_dims, **kwargs)

        self.base_kernel = base_kernel
        self.num_dims = num_dims
        if max_degree is None:
            self.max_degree = self.num_dims
        elif max_degree > self.num_dims:  # force cap on max_degree (silently)
            self.max_degree = self.num_dims
        else:
            self.max_degree = max_degree

        self.register_parameter(
            name="raw_outputscale", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, self.max_degree))
        )
        outputscale_constraint = Positive()
        self.register_constraint("raw_outputscale", outputscale_constraint)
        self.outputscale_constraint = outputscale_constraint
        self.outputscale = [1 / self.max_degree for _ in range(self.max_degree)]

    @property
    def outputscale(self):
        pass

    @outputscale.setter
    def outputscale(self, value):
        pass

    def _set_outputscale(self, value):
        pass

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """Forward proceeds by Newton-Girard formulae"""
        pass
