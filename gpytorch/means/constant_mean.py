#!/usr/bin/env python3

from __future__ import annotations

import warnings
from typing import Any

import torch

from ..constraints import Interval
from ..priors import Prior
from ..utils.warnings import OldVersionWarning
from .mean import Mean


def _ensure_updated_strategy_flag_set(
    state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
):
    pass


class ConstantMean(Mean):
    r"""
    A (non-zero) constant prior mean function, i.e.:

    .. math::
        \mu(\mathbf x) = C

    where :math:`C` is a learned constant.

    :param constant_prior: Prior for constant parameter :math:`C`.
    :type constant_prior: ~gpytorch.priors.Prior, optional
    :param constant_constraint: Constraint for constant parameter :math:`C`.
    :type constant_constraint: ~gpytorch.priors.Interval, optional
    :param batch_shape: The batch shape of the learned constant(s) (default: []).
    :type batch_shape: torch.Size, optional

    :var torch.Tensor constant: :math:`C` parameter
    """

    def __init__(
        self,
        constant_prior: Prior | None = None,
        constant_constraint: Interval | None = None,
        batch_shape: torch.Size = torch.Size(),
        **kwargs: Any,
    ):
        super().__init__()

        # Deprecated kwarg
        constant_prior_deprecated = kwargs.get("prior")
        if constant_prior_deprecated is not None:
            if constant_prior is None:  # Using the old kwarg for the constant_prior
                warnings.warn(
                    "The kwarg `prior` for ConstantMean has been renamed to `constant_prior`, and will be deprecated.",
                    DeprecationWarning,
                )
                constant_prior = constant_prior_deprecated
            else:  # Weird edge case where someone set both `prior` and `constant_prior`
                warnings.warn(
                    "You have set both the `constant_prior` and the deprecated `prior` arguments for ConstantMean. "
                    "`prior` is deprecated, and will be ignored.",
                    DeprecationWarning,
                )

        # Ensure that old versions of the model still load
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

        self.batch_shape = batch_shape
        self.register_parameter(name="raw_constant", parameter=torch.nn.Parameter(torch.zeros(batch_shape)))
        if constant_prior is not None:
            self.register_prior("mean_prior", constant_prior, self._constant_param, self._constant_closure)
        if constant_constraint is not None:
            self.register_constraint("raw_constant", constant_constraint)

    @property
    def constant(self):
        pass

    @constant.setter
    def constant(self, value):
        pass

    # We need a getter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_param(self, m):
        pass

    # We need a setter of this form so that we can pickle ConstantMean modules with a mean prior, see PR #1992
    def _constant_closure(self, m, value):
        pass

    def forward(self, input):
        pass
