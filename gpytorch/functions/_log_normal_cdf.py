#!/usr/bin/env python3

from __future__ import annotations

import math

import torch
from torch.autograd import Function
from torch.distributions import Normal


class LogNormalCDF(Function):
    @staticmethod
    def forward(ctx, z):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass
