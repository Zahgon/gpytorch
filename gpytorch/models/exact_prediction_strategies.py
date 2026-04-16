#!/usr/bin/env python3

import functools
import string
import warnings

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import (
    AddedDiagLinearOperator,
    BatchRepeatLinearOperator,
    ConstantMulLinearOperator,
    InterpolatedLinearOperator,
    LinearOperator,
    LowRankRootAddedDiagLinearOperator,
    MaskedLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    ZeroLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.interpolation import left_interp, left_t_interp
from torch import Tensor

from .. import settings
from ..distributions import MultitaskMultivariateNormal
from ..lazy import LazyEvaluatedKernelTensor
from ..utils.memoize import add_to_cache, cached, clear_cache_hook, pop_from_cache


def prediction_strategy(train_inputs, train_prior_dist, train_labels, likelihood):
    train_train_covar = train_prior_dist.lazy_covariance_matrix
    if isinstance(train_train_covar, LazyEvaluatedKernelTensor):
        cls = train_train_covar.kernel.prediction_strategy
    else:
        cls = DefaultPredictionStrategy
    return cls(train_inputs, train_prior_dist, train_labels, likelihood)


class DefaultPredictionStrategy:
    def __init__(
        self,
        train_inputs,
        train_prior_dist,
        train_labels,
        likelihood,
        root=None,
        inv_root=None,
    ):
        # Get training shape
        self._train_shape = train_prior_dist.event_shape

        # Flatten the training labels
        try:
            train_labels = train_labels.reshape(
                *train_labels.shape[: -len(self.train_shape)], self._train_shape.numel()
            )
        except RuntimeError:
            raise RuntimeError(
                "Flattening the training labels failed. The most common cause of this error is "
                + "that the shapes of the prior mean and the training labels are mismatched. "
                + f"The shape of the train targets is {train_labels.shape}, "
                + f"while the reported shape of the mean is {train_prior_dist.mean.shape}."
            )

        self.train_inputs = train_inputs
        self.train_prior_dist = train_prior_dist
        self.train_labels = train_labels
        self.likelihood = likelihood
        self._last_test_train_covar = None
        mvn = self.likelihood(train_prior_dist, train_inputs)
        self.lik_train_train_covar = mvn.lazy_covariance_matrix

        if root is not None:
            add_to_cache(
                self.lik_train_train_covar,
                "root_decomposition",
                RootLinearOperator(root),
            )

        if inv_root is not None:
            add_to_cache(
                self.lik_train_train_covar,
                "root_inv_decomposition",
                RootLinearOperator(inv_root),
            )

    def __deepcopy__(self, memo):
        # deepcopying prediction strategies of a model evaluated on inputs that require gradients fails
        # with RuntimeError (Only Tensors created explicitly by the user (graph leaves) support the deepcopy
        # protocol at the moment). Overwriting this method make sure that the prediction strategies of a
        # model are set to None upon deepcopying.
        pass

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        """
        Computes a cache for K_X*X (K_XX + sigma^2 I)^-1 K_X*X if possible. By default, this does no work and returns
        the first argument.

        Args:
            train_train_covar_inv_root (:obj:`torch.tensor`): a root of (K_XX + sigma^2 I)^-1
            test_train_covar (:obj:`torch.tensor`): the observed noise (from the likelihood)

        Returns
            A precomputed cache
        """
        pass

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        r"""
        Computes :math:`K_{X^{*}X} S` given a precomputed cache
        Where :math:`S` is a tensor such that :math:`SS^{\top} = (K_{XX} + \sigma^2 I)^{-1}`

        Args:
            precomputed_cache (:obj:`torch.tensor`): What was computed in _exact_predictive_covar_inv_quad_form_cache
            test_train_covar (:obj:`torch.tensor`): The observed noise (from the likelihood)

        Returns
            :obj:`~linear_operator.operators.LinearOperator`: :math:`K_{X^{*}X} S`
        """
        pass

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        """
        Returns a new PredictionStrategy that incorporates the specified inputs and targets as new training data.

        This method is primary responsible for updating the mean and covariance caches. To add fantasy data to a
        GP model, use the :meth:`~gpytorch.models.ExactGP.get_fantasy_model` method.

        Args:
            inputs (Tensor `b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`): Locations of fantasy
                observations.
            targets (Tensor `b1 x ... x bk x m` or `f x b1 x ... x bk x m`): Labels of fantasy observations.
            full_inputs (Tensor `b1 x ... x bk x n+m x d` or `f x b1 x ... x bk x n+m x d`): Training data
                concatenated with fantasy inputs
            full_targets (Tensor `b1 x ... x bk x n+m` or `f x b1 x ... x bk x n+m`): Training labels
                concatenated with fantasy labels.
            full_output (:class:`gpytorch.distributions.MultivariateNormal`): Prior called on full_inputs

        Returns:
            A `DefaultPredictionStrategy` model with `n + m` training examples, where the `m` fantasy examples have
            been added and all test-time caches have been updated.
        """
        pass

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        pass

    @property
    def mean_cache(self):
        pass

    @cached(name="mean_cache")
    def _mean_cache(self, nan_policy: str) -> Tensor:
        pass

    @property
    def num_train(self):
        pass

    @property
    def train_shape(self):
        pass

    def exact_prediction(
        self,
        test_mean: Tensor,
        test_test_covar: Tensor | LinearOperator,
        test_train_covar: Tensor | LinearOperator,
    ) -> tuple[Tensor, Tensor | LinearOperator]:
        """
        Computes the exact posterior distribution of the GP evaluated at test points.

        This method computes the posterior predictive mean and covariance using the
        standard GP conditioning formula.

        Note on batch dimensions: The test covariances (test_test_covar, test_train_covar)
        may have additional leading batch dimensions beyond the model's batch_shape.
        For example, additive models may pass covariances with shape
        (num_components, *batch_shape, n, n) to enable inference of individual additive
        components. The method will handle broadcasting appropriately.

        Note on masking: If observations contain NaN values that require masking, the
        masking should be applied to test_train_covar before calling this method.
        Override `_get_test_prior_mean_and_covariances` in your model to handle masking.

        :param test_mean: The test prior mean (shape: ... x num_test)
        :param test_test_covar: Covariance matrix between test inputs
            (shape: extra_batch x ... x num_test x num_test)
        :param test_train_covar: Covariance matrix between test and train inputs
            (shape: extra_batch x ... x num_test x num_train)
        :return: A tuple (posterior_predictive_mean, posterior_predictive_covar), with shapes
            (extra_batch x ... x num_test), and (extra_batch x ... x num_test x num_test)
        """
        pass

    def exact_predictive_mean(self, test_mean: Tensor, test_train_covar: LinearOperator) -> Tensor:
        """
        Computes the posterior predictive mean of a GP.

        Note: If observations contain NaN values that require masking, the masking
        should be applied to test_train_covar before calling this method. Override
        `_get_test_prior_mean_and_covariances` in your model to handle masking.

        :param Tensor test_mean: The test prior mean
        :param ~linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs (pre-masked if needed)
        :return: The predictive posterior mean of the test points
        """
        pass

    def exact_predictive_covar(
        self, test_test_covar: LinearOperator, test_train_covar: LinearOperator
    ) -> LinearOperator:
        """
        Computes the posterior predictive covariance of a GP

        :param ~linear_operator.operators.LinearOperator test_train_covar:
            Covariance matrix between test and train inputs
        :param ~linear_operator.operators.LinearOperator test_test_covar: Covariance matrix between test inputs
        :return: A LinearOperator representing the predictive posterior covariance of the test points
        """
        pass


class InterpolatedPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, uses_wiski=False):
        train_prior_dist = train_prior_dist.__class__(
            train_prior_dist.mean,
            train_prior_dist.lazy_covariance_matrix.evaluate_kernel(),
        )
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)
        self.uses_wiski = uses_wiski

    def _exact_predictive_covar_inv_quad_form_cache(self, train_train_covar_inv_root, test_train_covar):
        pass

    def _exact_predictive_covar_inv_quad_form_root(self, precomputed_cache, test_train_covar):
        # Here the precomputed cache represents K_UU W S,
        # where S S^T = (K_XX + sigma^2 I)^-1
        pass

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        r"""
        Implements the fantasy strategy described in https://arxiv.org/abs/2103.01454.
        """
        pass

    def prepare_dense_wmat(self, covar=None):
        # prepare the w matrix which is batch shape x m x n, where n = covar.shape[-2]
        pass

    @property
    @cached(name="interp_inner_prod")
    def interp_inner_prod(self):
        # the W'W cache
        pass

    @property
    @cached(name="interp_response_cache")
    def interp_response_cache(self):
        pass

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        pass

    @property
    @cached(name="fantasy_mean_cache")
    def fantasy_mean_cache(self):
        # first construct K_UU
        pass

    @property
    @cached(name="fantasy_covar_cache")
    def fantasy_covar_cache(self):
        pass

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Get inverse root
        pass

    def exact_prediction(
        self,
        test_mean: Tensor,
        test_test_covar: Tensor | LinearOperator,
        test_train_covar: Tensor | LinearOperator,
    ) -> tuple[Tensor, LinearOperator]:
        """
        Computes the posterior predictive distribution for interpolated (e.g. KISS) models.

        Unlike the default strategy, this method preserves the linear algebraic structure
        of the covariance matrices by avoiding calls to `to_dense()`, which is critical
        for the efficiency of KISS-GP models that rely on Kronecker and Toeplitz structure.

        :param test_mean: Prior mean at test points. Shape: ``(*batch_shape, num_test)`` or
            ``(*batch_shape, num_test, num_tasks)`` for multitask.
        :param test_test_covar: Prior covariance between test points. Shape:
            ``(*batch_shape, num_test, num_test)`` or
            ``(*batch_shape, num_test * num_tasks, num_test * num_tasks)`` for multitask.
        :param test_train_covar: Prior covariance between test and training points.
            Must be an InterpolatedLinearOperator to access interpolation indices/values.
            Shape: ``(*batch_shape, num_test, num_train)`` or
            ``(*batch_shape, num_test * num_tasks, num_train * num_tasks)`` for multitask.
        :return: A tuple of ``(predictive_mean, predictive_covar)`` where:
            - ``predictive_mean``: ``(*batch_shape, num_test)`` or
              ``(*batch_shape, num_test, num_tasks)``
            - ``predictive_covar``: LinearOperator with same shape as ``test_test_covar``
        """
        pass

    def exact_predictive_mean(self, test_mean, test_train_covar):
        pass

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        pass


class LinearPredictionStrategy(DefaultPredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood):
        train_prior_dist = train_prior_dist.__class__(
            train_prior_dist.mean,
            train_prior_dist.lazy_covariance_matrix.evaluate_kernel(),
        )
        super().__init__(train_inputs, train_prior_dist, train_labels, likelihood)

    def exact_prediction(
        self,
        test_mean: Tensor,
        test_test_covar: Tensor | LinearOperator,
        test_train_covar: Tensor | LinearOperator,
    ) -> tuple[Tensor, LinearOperator]:
        """
        Computes the exact moments of the posterior distribution of the GP
        evaluated on the test points.

        For low-rank kernels (e.g., LinearKernel, RFFKernel), the test_train_covar
        is not used because the test features can be extracted from test_test_covar.root
        and the training features are already baked into the mean_cache. This is an
        efficiency optimization using the Searle identity to work in d-dimensional
        feature space rather than n-dimensional data space.

        :param test_mean: The test prior mean (shape: ... x num_test)
        :param test_test_covar: Covariance matrix between test inputs
            (shape: ... x num_test x num_test)
        :param test_train_covar: Covariance matrix between test and train inputs
            (shape: ... x num_test x num_train). Not used for low-rank kernels.
        :return: A tuple (posterior_predictive_mean, posterior_predictive_covar), with shapes
            (... x num_test), and (... x num_test x num_test) respectively.
        """
        pass

    @property
    @cached(name="mean_covar_cache")
    def mean_covar_cache(self) -> tuple[Tensor, Tensor]:
        pass

    @property
    @cached(name="mean_cache")
    def mean_cache(self):
        # For finite-basis GPs with d < n features,
        # it is more efficient to compute mean and covar caches together (as they rely on the same
        # intermediate computations).
        pass

    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # For finite-basis GPs with d < n features,
        # it is more efficient to compute mean and covar caches together (as they rely on the same
        # intermediate computations).
        pass

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        r"""Implements fantasy observation updates for linear models.

        .. note::

            Unlike other fantasy strategies, we do not update the mean/covar caches; instead, we recompute them
            from scratch.

            The mean/covar caches essentially rely on the Cholesky factorization of the scatter matrix
            $\boldsymbol X^\top \boldsymbol X \in \mathbb R^{d \times d}$, where $d$ is the number of features.
            While we can update this Cholesky factorization using rank-one updates, we only obtain wall-clock
            speedups when the number of fantasy points is small (e.g. 1-2).
            With more fantasy points, a recomputation of the Cholesky factorization is faster than applying
            multiple sequential rank-one updates.

        :param Tensor inputs: The fantasy inputs
        :param Tensor targets: The fantasy targets
        :param List[Tensor] full_inputs: The full training inputs (including fantasies)
        :param Tensor full_targets: The full training targets (including fantasies)
        :param MultivariateNormal full_output: The output of the model on the full training data
        :return: A new prediction strategy incorporating the fantasy data
        """
        pass

    def exact_predictive_mean(self, *args, **kwargs):
        # For finite-basis GPs with d < n features,
        # it is more efficient to compute the posterior mean and covariance together
        # (as they both rely on multiplying the input features `x` by a vector/matrix),
        # and both rely on the same precomputation (i.e. pulling out the outputscale constant,
        # as in `exact_prediction`).
        # Therefore, we prevent users from calling these methods separately.
        raise RuntimeError("This method should not be called (use exact_prediction instead)!")

    def exact_predictive_covar(self, *args, **kwargs):
        # For finite-basis GPs with d < n features,
        # it is more efficient to compute the posterior mean and covariance together
        # (as they both rely on multiplying the input features `x` by a vector/matrix),
        # and both rely on the same precomputation (i.e. pulling out the outputscale constant,
        # as in `exact_prediction`).
        # Therefore, we prevent users from calling these methods separately.
        raise RuntimeError("This method should not be called (use exact_prediction instead)!")


class SGPRPredictionStrategy(DefaultPredictionStrategy):
    @property
    @cached(name="covar_cache")
    def covar_cache(self):
        # Here, the covar_cache is going to be K_{UU}^{-1/2} K_{UX}( K_{XX} + \sigma^2 I )^{-1} K_{XU} K_{UU}^{-1/2}
        # This is easily computed using Woodbury
        # K_{XX} + \sigma^2 I = R R^T + \sigma^2 I
        #                     = \sigma^{-2} ( I - \sigma^{-2} R (I + \sigma^{-2} R^T R)^{-1} R^T  )
        pass

    def get_fantasy_strategy(self, inputs, targets, full_inputs, full_targets, full_output, **kwargs):
        raise NotImplementedError(
            "Fantasy observation updates not yet supported for models using SGPRPredictionStrategy"
        )

    def exact_prediction(
        self,
        test_mean: Tensor,
        test_test_covar: Tensor | LinearOperator,
        test_train_covar: Tensor | LinearOperator,
    ) -> tuple[Tensor, Tensor | LinearOperator]:
        pass

    def exact_predictive_covar(self, test_test_covar, test_train_covar):
        pass
