#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import (
    ConstantDiagLinearOperator,
    DiagLinearOperator,
    KroneckerProductDiagLinearOperator,
    KroneckerProductLinearOperator,
    LinearOperator,
    RootLinearOperator,
)
from torch import Tensor
from torch.distributions import Normal

from ..constraints import GreaterThan, Interval
from ..distributions import base_distributions, MultitaskMultivariateNormal
from ..lazy import LazyEvaluatedKernelTensor
from ..likelihoods import _GaussianLikelihoodBase, Likelihood
from ..priors import Prior
from .noise_models import FixedGaussianNoise, Noise


class _MultitaskGaussianLikelihoodBase(_GaussianLikelihoodBase):
    r"""
    Base class for multi-task Gaussian Likelihoods, supporting general heteroskedastic noise models.

    :param num_tasks: Number of tasks.
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_correlation_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    """

    def __init__(
        self,
        num_tasks: int,
        noise_covar: Noise | FixedGaussianNoise,
        rank: int = 0,
        task_correlation_prior: Prior | None = None,
        batch_shape: torch.Size = torch.Size(),
    ) -> None:
        super().__init__(noise_covar=noise_covar)
        if rank != 0:
            if rank > num_tasks:
                raise ValueError(f"Cannot have rank ({rank}) greater than num_tasks ({num_tasks})")
            tidcs = torch.tril_indices(num_tasks, rank, dtype=torch.long)
            self.tidcs: Tensor = tidcs[:, 1:]  # (1, 1) must be 1.0, no need to parameterize this
            task_noise_corr = torch.randn(*batch_shape, self.tidcs.size(-1))
            self.register_parameter("task_noise_corr", torch.nn.Parameter(task_noise_corr))
            if task_correlation_prior is not None:
                self.register_prior(
                    "MultitaskErrorCorrelationPrior", task_correlation_prior, lambda m: m._eval_corr_matrix
                )
        elif task_correlation_prior is not None:
            raise ValueError("Can only specify task_correlation_prior if rank>0")
        self.num_tasks = num_tasks
        self.rank = rank

    def _eval_corr_matrix(self) -> Tensor:
        pass

    def marginal(
        self, function_dist: MultitaskMultivariateNormal, *params: Any, **kwargs: Any
    ) -> MultitaskMultivariateNormal:  # pyre-ignore[14]
        r"""
        If :math:`\text{rank} = 0`, adds the task noises to the diagonal of the
        covariance matrix of the supplied
        :obj:`~gpytorch.distributions.MultivariateNormal` or
        :obj:`~gpytorch.distributions.MultitaskMultivariateNormal`.  Otherwise,
        adds a rank `rank` covariance matrix to it.

        To accomplish this, we form a new
        :obj:`~linear_operator.operators.KroneckerProductLinearOperator`
        between :math:`I_{n}`, an identity matrix with size equal to the data
        and a (not necessarily diagonal) matrix containing the task noises
        :math:`D_{t}`.

        We also incorporate a shared `noise` parameter from the base
        :class:`gpytorch.likelihoods.GaussianLikelihood` that we extend.

        The final covariance matrix after this method is then
        :math:`\mathbf K + \mathbf D_{t} \otimes \mathbf I_{n} + \sigma^{2} \mathbf I_{nt}`.

        :param function_dist: Random variable whose covariance
            matrix is a :obj:`~linear_operator.operators.LinearOperator` we intend to augment.
        :rtype: `gpytorch.distributions.MultitaskMultivariateNormal`:
        :return: A new random variable whose covariance matrix is a
            :obj:`~linear_operator.operators.LinearOperator` with
            :math:`\mathbf D_{t} \otimes \mathbf I_{n}` and :math:`\sigma^{2} \mathbf I_{nt}` added.
        """
        pass

    def _shaped_noise_covar(
        self, shape: torch.Size, add_noise: bool | None = True, interleaved: bool = True, *params: Any, **kwargs: Any
    ) -> LinearOperator:
        pass

    def forward(self, function_samples: Tensor, *params: Any, **kwargs: Any) -> Normal:
        pass


class MultitaskGaussianLikelihood(_MultitaskGaussianLikelihoodBase):
    r"""
    A convenient extension of the :class:`~gpytorch.likelihoods.GaussianLikelihood` to the multitask setting that allows
    for a full cross-task covariance structure for the noise. The fitted covariance matrix has rank `rank`.
    If a strictly diagonal task noise covariance matrix is desired, then rank=0 should be set. (This option still
    allows for a different `noise` parameter for each task.)

    Like the Gaussian likelihood, this object can be used with exact inference.

    .. note::
        At least one of :attr:`has_global_noise` or :attr:`has_task_noise` should be specified.

    .. note::
        MultittaskGaussianLikelihood has an analytic marginal distribution.

    :param num_tasks: Number of tasks.
    :param noise_covar: A model for the noise covariance. This can be a simple homoskedastic noise model, or a GP
        that is to be fitted on the observed measurement errors.
    :param rank: The rank of the task noise covariance matrix to fit. If `rank`
        is set to 0, then a diagonal covariance matrix is fit.
    :param task_prior: Prior to use over the task noise correlation
        matrix. Only used when :math:`\text{rank} > 0`.
    :param batch_shape: Number of batches.
    :param has_global_noise: Whether to include a :math:`\sigma^2 \mathbf I_{nt}` term in the noise model.
    :param has_task_noise: Whether to include task-specific noise terms, which add
        :math:`\mathbf I_n \otimes \mathbf D_T` into the noise model.

    :ivar torch.Tensor task_noise_covar: The inter-task noise covariance matrix
    :ivar torch.Tensor task_noises: (Optional) task specific noise variances (added onto the `task_noise_covar`)
    :ivar torch.Tensor noise: (Optional) global noise variance (added onto the `task_noise_covar`)
    """

    def __init__(
        self,
        num_tasks: int,
        rank: int = 0,
        batch_shape: torch.Size = torch.Size(),
        task_prior: Prior | None = None,
        noise_prior: Prior | None = None,
        noise_constraint: Interval | None = None,
        has_global_noise: bool = True,
        has_task_noise: bool = True,
    ) -> None:
        super(Likelihood, self).__init__()  # pyre-ignore[20]
        if noise_constraint is None:
            noise_constraint = GreaterThan(1e-4)

        if not has_task_noise and not has_global_noise:
            raise ValueError(
                "At least one of has_task_noise or has_global_noise must be specified. "
                "Attempting to specify a likelihood that has no noise terms."
            )

        if has_task_noise:
            if rank == 0:
                self.register_parameter(
                    name="raw_task_noises", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_tasks))
                )
                self.register_constraint("raw_task_noises", noise_constraint)
                if noise_prior is not None:
                    self.register_prior("raw_task_noises_prior", noise_prior, lambda m: m.task_noises)
                if task_prior is not None:
                    raise RuntimeError("Cannot set a `task_prior` if rank=0")
            else:
                self.register_parameter(
                    name="task_noise_covar_factor",
                    parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_tasks, rank)),
                )
                if task_prior is not None:
                    self.register_prior("MultitaskErrorCovariancePrior", task_prior, lambda m: m._eval_covar_matrix)
        self.num_tasks = num_tasks
        self.rank = rank

        if has_global_noise:
            self.register_parameter(name="raw_noise", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
            self.register_constraint("raw_noise", noise_constraint)
            if noise_prior is not None:
                self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

        self.has_global_noise = has_global_noise
        self.has_task_noise = has_task_noise

    @property
    def noise(self) -> Tensor | None:
        pass

    @noise.setter
    def noise(self, value: float | Tensor) -> None:
        pass

    @property
    def task_noises(self) -> Tensor | None:
        pass

    @task_noises.setter
    def task_noises(self, value: float | Tensor) -> None:
        pass

    def _set_noise(self, value: float | Tensor) -> None:
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    def _set_task_noises(self, value: float | Tensor) -> None:
        pass

    @property
    def task_noise_covar(self) -> Tensor:
        pass

    @task_noise_covar.setter
    def task_noise_covar(self, value: Tensor) -> None:
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        pass

    def _eval_covar_matrix(self) -> Tensor:
        covar_factor = self.task_noise_covar_factor
        noise = self.noise
        D = noise * torch.eye(self.num_tasks, dtype=noise.dtype, device=noise.device)  # pyre-fixme[16]
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(
        self, function_dist: MultitaskMultivariateNormal, *args: Any, **kwargs: Any
    ) -> MultitaskMultivariateNormal:
        r"""
        :return: Analytic marginal :math:`p(\mathbf y)`.
        """
        pass
