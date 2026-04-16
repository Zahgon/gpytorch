from __future__ import annotations

import torch


class RBFCovariance(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, lengthscale, sq_dist_func):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        pass
