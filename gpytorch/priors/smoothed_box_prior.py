#!/usr/bin/env python3

from __future__ import annotations

import math
from numbers import Number

import torch
from torch.distributions import constraints
from torch.distributions.utils import broadcast_all
from torch.nn import Module as TModule

from .prior import Prior
from .torch_priors import NormalPrior


class SmoothedBoxPrior(Prior):
    r"""A smoothed approximation of a uniform prior.

    Has full support on the reals and is differentiable everywhere.

    .. math::

        \begin{equation*}
            B = {x: a_i <= x_i <= b_i}
            d(x, B) = min_{x' in B} |x - x'|
            pdf(x) \sim exp(- d(x, B)**2 / sqrt(2 * sigma^2))
        \end{equation*}

    """

    arg_constraints = {"sigma": constraints.positive, "a": constraints.real, "b": constraints.real}
    support = constraints.real
    has_rsample = True
    _validate_args = True

    def __init__(self, a, b, sigma=0.01, validate_args=False, transform=None):
        TModule.__init__(self)
        _a = torch.tensor(float(a)) if isinstance(a, Number) else a
        _a = _a.view(-1) if _a.dim() < 1 else _a
        _a, _b, _sigma = broadcast_all(_a, b, sigma)
        if not torch.all(constraints.less_than(_b).check(_a)):
            raise ValueError("must have that a < b (element-wise)")
        # TODO: Proper argument validation including broadcasting
        batch_shape, event_shape = _a.shape[:-1], _a.shape[-1:]
        # need to assign values before registering as buffers to make argument validation work
        self.a, self.b, self.sigma = _a, _b, _sigma
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        # now need to delete to be able to register buffer
        del self.a, self.b, self.sigma
        self.register_buffer("a", _a)
        self.register_buffer("b", _b)
        self.register_buffer("sigma", _sigma.clone())
        self.tails = NormalPrior(torch.zeros_like(_a), _sigma, validate_args=validate_args)
        self._transform = transform

    @property
    def _c(self):
        pass

    @property
    def _r(self):
        pass

    @property
    def _M(self):
        # normalization factor to make this a probability distribution
        pass

    def log_prob(self, x):
        return self._log_prob(self.transform(x))

    def _log_prob(self, x):
        # x = "distances from box`"
        X = ((x - self._c).abs_() - self._r).clamp(min=0)
        return (self.tails.log_prob(X) - self._M).sum(-1)

    def rsample(self, sample_shape=torch.Size()):
        pass
