from __future__ import annotations

import math

import torch


class MaternCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, nu, dist_func):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass
