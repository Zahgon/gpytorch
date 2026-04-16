#!/usr/bin/env python3

from __future__ import annotations

import copy
import math

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    DiagLinearOperator,
    LowRankRootAddedDiagLinearOperator,
    LowRankRootLinearOperator,
    MatmulLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import Tensor

from .. import settings
from ..distributions import MultivariateNormal
from ..likelihoods import Likelihood
from ..mlls import InducingPointKernelAddedLossTerm
from ..models import exact_prediction_strategies
from .kernel import Kernel


class InducingPointKernel(Kernel):
    def __init__(
        self,
        base_kernel: Kernel,
        inducing_points: Tensor,
        likelihood: Likelihood,
        active_dims: tuple[int, ...] | None = None,
    ):
        super().__init__(active_dims=active_dims)
        self.base_kernel = base_kernel
        self.likelihood = likelihood

        if inducing_points.ndimension() == 1:
            inducing_points = inducing_points.unsqueeze(-1)

        self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        self.register_added_loss_term("inducing_point_loss_term")

    def _clear_cache(self):
        pass

    @property
    def _inducing_mat(self):
        pass

    @property
    def _inducing_inv_root(self):
        pass

    def _get_covariance(self, x1, x2):
        pass

    def _covar_diag(self, inputs):
        pass

    def forward(self, x1, x2, diag=False, **kwargs):
        pass

    def num_outputs_per_input(self, x1, x2):
        pass

    def __deepcopy__(self, memo):
        replace_inv_root = False
        replace_kernel_mat = False

        if hasattr(self, "_cached_kernel_inv_root"):
            replace_inv_root = True
            kernel_inv_root = self._cached_kernel_inv_root
        if hasattr(self, "_cached_kernel_mat"):
            replace_kernel_mat = True
            kernel_mat = self._cached_kernel_mat

        cp = self.__class__(
            base_kernel=copy.deepcopy(self.base_kernel),
            inducing_points=copy.deepcopy(self.inducing_points),
            likelihood=self.likelihood,
            active_dims=self.active_dims,
        )

        if replace_inv_root:
            cp._cached_kernel_inv_root = kernel_inv_root

        if replace_kernel_mat:
            cp._cached_kernel_mat = kernel_mat

        return cp

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        # Allow for fast variances
        return exact_prediction_strategies.SGPRPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
