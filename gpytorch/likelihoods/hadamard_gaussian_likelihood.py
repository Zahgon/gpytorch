#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import torch

from linear_operator.operators import DiagLinearOperator, LinearOperator

from ..constraints import Interval

from ..distributions import base_distributions, MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from ..priors import Prior
from .noise_models import MultitaskHomoskedasticNoise


def _check_task_indices(task_idcs: torch.Tensor, base_shape: torch.Size) -> None:
    r"""
    Check that the task indices have valid dtype, and are the correct shape.

    This is agnostic to additional dimensions in base_shape, which are
    added in the likelihood's forward method.

    Args:
        task_idcs: Tensor of task indices.
        base_shape: Shape of the data.
    """
    pass


class HadamardGaussianLikelihood(_GaussianLikelihoodBase):
    r"""
    Likelihood for input-wise homoskedastic and task-wise heteroskedastic noise,
    i.e. we learn a different (constant) noise level for each fidelity.

    Args:
        num_of_tasks: Number of tasks in the multi-output GP.
        noise_prior: Prior for the noise. This can be multi-dimensional to apply
            different priors to each task, however all tasks must have the same
            type of prior.
        noise_constraint: Constraint on the noise value.
        batch_shape: The batch shape of the learned noise parameter (default: []).
        task_feature_index: The index of the task feature in the input data (default: None).
    """

    def __init__(
        self,
        num_tasks: int,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        batch_shape: torch.Size = torch.Size(),
        task_feature_index: int | None = None,
        **kwargs,
    ):
        noise_covar = MultitaskHomoskedasticNoise(
            num_tasks=num_tasks,
            noise_prior=noise_prior,
            noise_constraint=noise_constraint,
            batch_shape=batch_shape,
        )
        self.num_tasks = num_tasks
        self.task_feature_index = task_feature_index
        super().__init__(noise_covar=noise_covar, **kwargs)

    @property
    def noise(self) -> torch.Tensor:
        pass

    @noise.setter
    def noise(self, value: torch.Tensor) -> None:
        pass

    @property
    def raw_noise(self) -> torch.Tensor:
        pass

    @raw_noise.setter
    def raw_noise(self, value: torch.Tensor) -> None:
        pass

    def _shaped_noise_covar(self, base_shape: torch.Size, *params: Any, **kwargs: Any) -> LinearOperator:
        # params contains input data, shape (*task_batch_shape, num_data, d)
        pass

    def forward(
        self,
        function_samples: torch.Tensor,
        *params: Any,
        **kwargs: Any,
    ) -> base_distributions.Normal:
        pass

    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        pass
