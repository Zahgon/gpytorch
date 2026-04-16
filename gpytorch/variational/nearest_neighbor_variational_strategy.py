#!/usr/bin/env python3

from __future__ import annotations

from typing import Any

import torch
from linear_operator import to_dense
from linear_operator.operators import DiagLinearOperator, LinearOperator, TriangularLinearOperator
from linear_operator.utils.cholesky import psd_safe_cholesky
from torch import LongTensor, Tensor

from ..distributions import MultivariateNormal
from ..models import ApproximateGP, ExactGP
from ..module import Module
from ..utils.errors import CachingError
from ..utils.memoize import add_to_cache, cached, pop_from_cache
from ..utils.nearest_neighbors import NNUtil
from ._variational_distribution import _VariationalDistribution
from .mean_field_variational_distribution import MeanFieldVariationalDistribution
from .unwhitened_variational_strategy import UnwhitenedVariationalStrategy


class NNVariationalStrategy(UnwhitenedVariationalStrategy):
    r"""
    This strategy sets all inducing point locations to observed inputs,
    and employs a :math:`k`-nearest-neighbor approximation. It was introduced as the
    `Variational Nearest Neighbor Gaussian Processes (VNNGP)` in `Wu et al (2022)`_.
    See the `VNNGP tutorial`_ for an example.

    VNNGP assumes a k-nearest-neighbor generative process for inducing points :math:`\mathbf u`,
    :math:`\mathbf q(\mathbf u) = \prod_{j=1}^M q(u_j | \mathbf u_{n(j)})`
    where :math:`n(j)` denotes the indices of :math:`k` nearest neighbors for :math:`u_j` among
    :math:`u_1, \cdots, u_{j-1}`. For any test observation :math:`\mathbf f`,
    VNNGP makes predictive inference conditioned on its :math:`k` nearest inducing points
    :math:`\mathbf u_{n(f)}`, i.e. :math:`p(f|\mathbf u_{n(f)})`.

    VNNGP's objective factorizes over inducing points and observations, making stochastic optimization over both
    immediately available. After a one-time cost of computing the :math:`k`-nearest neighbor structure,
    the training and inference complexity is :math:`O(k^3)`.
    Since VNNGP uses observations as inducing points, it is a user choice to either (1)
    use the same mini-batch of inducing points and observations (recommended),
    or (2) use different mini-batches of inducing points and observations. See the `VNNGP tutorial`_ for
    implementation and comparison.


    .. note::

        The current implementation only supports :obj:`~gpytorch.variational.MeanFieldVariationalDistribution`.

        We recommend installing the `faiss`_ library (requiring separate package installment)
        for nearest neighbor search, which is significantly faster than the `scikit-learn` nearest neighbor search.
        GPyTorch will automatically use `faiss` if it is installed, but will revert to `scikit-learn` otherwise.

        Different inducing point orderings will produce in different nearest neighbor approximations.


    :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param k: Number of nearest neighbors.
    :param training_batch_size: The number of data points that will be in the training batch size.
    :param jitter_val: Amount of diagonal jitter to add for covariance matrix numerical stability.
    :param compute_full_kl: Whether to compute full kl divergence or stochastic estimate.

    .. _Wu et al (2022):
        https://arxiv.org/pdf/2202.01694.pdf
    .. _VNNGP tutorial:
        examples/04_Variational_and_Approximate_GPs/VNNGP.html
    .. _faiss:
        https://github.com/facebookresearch/faiss
    """

    def __init__(
        self,
        model: ApproximateGP,
        inducing_points: Tensor,  # shape: (..., M, D)
        variational_distribution: _VariationalDistribution,  # shape: (..., M)
        k: int,
        training_batch_size: int | None = None,
        jitter_val: float | None = 1e-3,
        compute_full_kl: bool | None = False,
    ):
        assert isinstance(
            variational_distribution, MeanFieldVariationalDistribution
        ), "Currently, NNVariationalStrategy only supports MeanFieldVariationalDistribution."

        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations=False, jitter_val=jitter_val
        )

        # Model
        object.__setattr__(self, "model", model)

        self.inducing_points = inducing_points
        self.M, self.D = inducing_points.shape[-2:]
        self.k = k
        assert self.k < self.M, (
            f"Number of nearest neighbors k must be smaller than the number of inducing points, "
            f"but got k = {k}, M = {self.M}."
        )

        self._inducing_batch_shape: torch.Size = inducing_points.shape[:-2]
        self._model_batch_shape: torch.Size = self._variational_distribution.variational_mean.shape[:-1]
        self._batch_shape: torch.Size = torch.broadcast_shapes(self._inducing_batch_shape, self._model_batch_shape)

        self.nn_util: NNUtil = NNUtil(
            k, dim=self.D, batch_shape=self._inducing_batch_shape, device=inducing_points.device
        )
        self._compute_nn()
        # otherwise, no nearest neighbor approximation is used

        self.training_batch_size = training_batch_size if training_batch_size is not None else self.M
        self._set_training_iterator()

        self.compute_full_kl = compute_full_kl

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:  # shape: (..., M)
        pass

    def _cholesky_factor(
        self,
        induc_induc_covar: LinearOperator,  # shape: (..., M, M)
    ) -> TriangularLinearOperator:  # shape: (..., M, M)
        # Uncached version
        pass

    def __call__(
        self,
        x: Tensor,  # shape: (..., N, D)
        prior: bool = False,
        diag: bool = True,
        **kwargs: Any,
    ) -> MultivariateNormal:  # shape: (..., N)
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        if x is not None:
            # Make sure x and inducing points have the same batch shape
            if not (self.inducing_points.shape[:-2] == x.shape[:-2]):
                try:
                    x = x.expand(*self.inducing_points.shape[:-2], *x.shape[-2:]).contiguous()
                except RuntimeError:
                    raise RuntimeError(
                        f"x batch shape must match or broadcast with the inducing points' batch shape, "
                        f"but got x batch shape = {x.shape[:-2]}, "
                        f"inducing points batch shape = {self.inducing_points.shape[:-2]}."
                    )

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()

            # (Maybe) initialize variational distribution
            if not self.variational_params_initialized.item():
                prior_dist = self.prior_distribution
                self._variational_distribution.variational_mean.data.copy_(prior_dist.mean)
                self._variational_distribution.variational_mean.data.add_(
                    torch.randn_like(prior_dist.mean), alpha=self._variational_distribution.mean_init_std
                )
                # initialize with a small variational stddev for quicker conv. of kl divergence
                self._variational_distribution._variational_stddev.data.copy_(torch.tensor(1e-2))
                self.variational_params_initialized.fill_(1)

            return self.forward(
                x, self.inducing_points, inducing_values=None, variational_inducing_covar=None, **kwargs
            )
        else:
            # Ensure inducing_points and x are the same size
            inducing_points = self.inducing_points
            return self.forward(x, inducing_points, inducing_values=None, variational_inducing_covar=None, **kwargs)

    def forward(
        self,
        x: Tensor,  # shape: (..., N, D)
        inducing_points: Tensor,  # shape: (..., M, D)
        inducing_values: Tensor,  # shape: (..., M)
        variational_inducing_covar: LinearOperator | None = None,  # shape: (..., M, M)
        diag: bool = True,
        **kwargs: Any,
    ) -> MultivariateNormal:  # shape: (..., N)
        # TODO: This method needs to return the full covariance in eval mode, not just the predictive variance.
        # TODO: Use `diag` to control when to compute the variance vs. covariance in train mode.
        pass

    def get_fantasy_model(
        self,
        inputs: Tensor,  # shape: (..., N, D)
        targets: Tensor,  # shape: (..., N)
        mean_module: Module | None = None,
        covar_module: Module | None = None,
        **kwargs,
    ) -> ExactGP:
        raise NotImplementedError(
            f"No fantasy model support for {self.__class__.__name__}. "
            "Only VariationalStrategy and UnwhitenedVariationalStrategy are currently supported."
        )

    def _set_training_iterator(self) -> None:
        self._training_indices_iter = 0
        if self.training_batch_size == self.M:
            self._training_indices_iterator = (torch.arange(self.M, device=self.inducing_points.device),)
        else:
            # The first training batch always contains the first k inducing points
            # This is because computing the KL divergence for the first k inducing points is special-cased
            # (since the first k inducing points have < k neighbors)
            # Note that there is a special function _firstk_kl_helper for this
            training_indices = torch.randperm(self.M - self.k, device=self.inducing_points.device) + self.k
            self._training_indices_iterator = (torch.arange(self.k),) + training_indices.split(self.training_batch_size)
        self._total_training_batches = len(self._training_indices_iterator)

    def _get_training_indices(self) -> LongTensor:
        pass

    def _firstk_kl_helper(self) -> Tensor:  # shape: (...)
        # Compute the KL divergence for first k inducing points
        pass

    def _stochastic_kl_helper(
        self,
        kl_indices: Tensor,  # shape: (n_batch,)
    ) -> Tensor:  # shape: (...)
        # Compute the KL divergence for a mini batch of the rest M-k inducing points
        # See paper appendix for kl breakdown
        pass

    def _kl_divergence(
        self, kl_indices: LongTensor | None = None, batch_size: int | None = None
    ) -> Tensor:  # shape: (...)
        pass

    def kl_divergence(self) -> Tensor:  # shape: (...)
        try:
            return pop_from_cache(self, "kl_divergence_memo")
        except CachingError:
            raise RuntimeError("KL Divergence of variational strategy was called before nearest neighbors were set.")

    def _compute_nn(self) -> NNVariationalStrategy:
        with torch.no_grad():
            inducing_points_fl = self.inducing_points.data.float()
            self.nn_util.set_nn_idx(inducing_points_fl)
            self.nn_xinduce_idx = self.nn_util.build_sequential_nn_idx(inducing_points_fl)
            #  shape (*_inducing_batch_shape, M-k, k)
        return self
