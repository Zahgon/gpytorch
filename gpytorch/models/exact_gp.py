#!/usr/bin/env python3

from __future__ import annotations

import warnings

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import Tensor

from gpytorch.distributions import Distribution

from .. import settings
from ..distributions import MultitaskMultivariateNormal, MultivariateNormal
from ..likelihoods import _GaussianLikelihoodBase
from ..utils.warnings import GPInputWarning
from .exact_prediction_strategies import prediction_strategy
from .gp import GP


class ExactGP(GP):
    r"""
    The base class for any Gaussian process latent function to be used in conjunction
    with exact inference.

    :param torch.Tensor train_inputs: (size n x d) The training features :math:`\mathbf X`.
    :param torch.Tensor train_targets: (size n) The training targets :math:`\mathbf y`.
    :param ~gpytorch.likelihoods.GaussianLikelihood likelihood: The Gaussian likelihood that defines
        the observational distribution. Since we're using exact inference, the likelihood must be Gaussian.

    The :meth:`forward` function should describe how to compute the prior latent distribution
    on a given input. Typically, this will involve a mean and kernel function.
    The result must be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Calling this model will return the posterior of the latent Gaussian process when conditioned
    on the training data. The output will be a :obj:`~gpytorch.distributions.MultivariateNormal`.

    Example:
        >>> class MyGP(gpytorch.models.ExactGP):
        >>>     def __init__(self, train_x, train_y, likelihood):
        >>>         super().__init__(train_x, train_y, likelihood)
        >>>         self.mean_module = gpytorch.means.ZeroMean()
        >>>         self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        >>>
        >>>     def forward(self, x):
        >>>         mean = self.mean_module(x)
        >>>         covar = self.covar_module(x)
        >>>         return gpytorch.distributions.MultivariateNormal(mean, covar)
        >>>
        >>> # train_x = ...; train_y = ...
        >>> likelihood = gpytorch.likelihoods.GaussianLikelihood()
        >>> model = MyGP(train_x, train_y, likelihood)
        >>>
        >>> # test_x = ...;
        >>> model(test_x)  # Returns the GP latent function at test_x
        >>> likelihood(model(test_x))  # Returns the (approximate) predictive posterior distribution at test_x
    """

    def __init__(
        self,
        train_inputs: Tensor | Iterable[Tensor] | None,
        train_targets: Tensor | None,
        likelihood: _GaussianLikelihoodBase,
    ):
        if train_inputs is not None and isinstance(train_inputs, Tensor):
            train_inputs = (train_inputs,)
        if train_inputs is not None and not all(isinstance(train_input, Tensor) for train_input in train_inputs):
            raise RuntimeError("Train inputs must be a tensor, or a list/tuple of tensors")
        if not isinstance(likelihood, _GaussianLikelihoodBase):
            raise RuntimeError("ExactGP can only handle Gaussian likelihoods")

        super().__init__()
        if train_inputs is not None:
            self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
            self.train_targets = train_targets
        else:
            self.train_inputs = None
            self.train_targets = None
        self.likelihood = likelihood

        self.prediction_strategy = None

    @property
    def train_targets(self) -> tuple[Tensor] | None:
        pass

    @train_targets.setter
    def train_targets(self, value: Tensor | None) -> None:
        pass

    def _apply(self, fn):
        if self.train_inputs is not None:
            self.train_inputs = tuple(fn(train_input) for train_input in self.train_inputs)
            self.train_targets = fn(self.train_targets)
        return super()._apply(fn)

    def _clear_cache(self) -> None:
        # The precomputed caches from test time live in prediction_strategy
        pass

    def local_load_samples(self, samples_dict, memo, prefix):
        """
        Replace the model's learned hyperparameters with samples from a posterior distribution.
        """
        pass

    def set_train_data(
        self, inputs: Tensor | Iterable[Tensor] | None = None, targets: Tensor | None = None, strict: bool = True
    ) -> None:
        """
        Set training data (does not re-fit model hyper-parameters).

        :param inputs: The new training inputs.
        :param targets: The new training targets.
        :param strict: If `True`, the new inputs and targets must have the same shape,
            dtype, and device as the current inputs and targets. Otherwise, any
            shape/dtype/device are allowed.
        """
        pass

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        Using this method is more efficient than updating with `set_train_data` when the number of inputs is relatively
        small, because any computed test-time caches will be updated in linear time rather than computed from scratch.

        .. note::
            If `targets` is a batch (e.g. `b x m`), then the GP returned from this method will be a batch mode GP.
            If `inputs` is of the same (or lesser) dimension as `targets`, then it is assumed that the fantasy points
            are the same for each target batch.

        :param torch.Tensor inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param torch.Tensor targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :return: An `ExactGP` model with `n + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated.
        :rtype: ~gpytorch.models.ExactGP
        """
        pass

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if settings.debug.on():
                if not all(
                    torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs, strict=True)
                ):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super().__call__(*full_inputs, **kwargs)
            if settings.debug.on():
                if not isinstance(full_output, MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if settings.debug.on():
                if all(
                    torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs, strict=True)
                ):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        GPInputWarning,
                    )

            # Get the terms that only depend on training data
            if self.prediction_strategy is None:
                train_output = self._get_train_prior_distribution(train_inputs, **kwargs)

                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )
            (
                test_mean,
                test_test_covar,
                test_train_covar,
                batch_shape,
                test_shape,
                posterior_class,
            ) = self._get_test_prior_mean_and_covariances(train_inputs=train_inputs, test_inputs=inputs, **kwargs)
            # Make the prediction
            with settings.cg_tolerance(settings.eval_cg_tolerance.value()):
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(
                    test_mean=test_mean,
                    test_test_covar=test_test_covar,
                    test_train_covar=test_train_covar,
                )

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return posterior_class(predictive_mean, predictive_covar)

    def _get_train_prior_distribution(
        self,
        train_inputs: Iterable[Tensor],
        **kwargs,
    ) -> MultivariateNormal:
        """Computes the prior distribution on the training set.

        Override this method to customize train-train covariance computation.

        Args:
            train_inputs: The inputs in the training set.
            kwargs: Additional keyword arguments passed to the model's forward method.

        Returns:
            The prior distribution evaluated on the training set.
        """
        pass

    def _get_test_prior_mean_and_covariances(
        self,
        train_inputs: Iterable[Tensor],
        test_inputs: Iterable[Tensor],
        **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor, torch.Size, torch.Size, type[Distribution]]:
        """Computes the prior mean and covariances on the test set.

        Override this method to customize test-set covariance computations, e.g.,
        for models with partial observations or per-component additive inference.

        The returned covariances may have additional leading batch dimensions
        (e.g., for additive component-wise inference). The prediction strategy
        handles broadcasting with the train-train covariance.

        Note: This method is efficient even when test_inputs overlaps with
        train_inputs. Slicing the lazy joint covariance only evaluates
        K(test, [train||test]); K(train, train) is never computed.

        Args:
            train_inputs: The training inputs.
            test_inputs: The test inputs.
            kwargs: Additional keyword arguments passed to the model's forward.

        Returns:
            A tuple of (test_mean, test_test_covar, test_train_covar, batch_shape,
            test_shape, posterior_class).
        """
        pass
