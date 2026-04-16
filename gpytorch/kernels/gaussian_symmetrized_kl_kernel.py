#!/usr/bin/env python3

from __future__ import annotations

# from torch.distributions import Normal, kl_divergence
from .distributional_input_kernel import DistributionalInputKernel


def _symmetrized_kl(dist1, dist2, eps=1e-8):
    """
    Symmetrized KL distance between two Gaussian distributions. We assume that
    the first half of the distribution tensors are the mean, and the second half
    are the log variances.
    Args:
        dist1 (torch.Tensor) has shapes batch x n x dimensions. The first half
            of the last dimensions are the means, while the second half are the log-variances.
        dist2 (torch.Tensor) has shapes batch x n x dimensions. The first half
            of the last dimensions are the means, while the second half are the log-variances.
        eps (float) jitter term for the noise variance
    """
    pass


class GaussianSymmetrizedKLKernel(DistributionalInputKernel):
    r"""
    Computes a kernel based on the symmetrized KL divergence, assuming that two Gaussian
    distributions are inputted. Inputs are assumed to be `batch x N x 2d` tensors where `d` is the
    dimension of the distribution. The first `d` dimensions are the mean parameters of the
    `batch x N` distributions, while the second `d` dimensions are the log variances.

    Original citation is Moreno et al, '04
    (https://papers.nips.cc/paper/2351-a-kullback-leibler-divergence-based-kernel-for-svm-\
    classification-in-multimedia-applications.pdf) for the symmetrized KL divergence kernel between
    two Gaussian distributions.
    """

    def __init__(self, **kwargs):
        distance_function = _symmetrized_kl
        super().__init__(distance_function=distance_function, **kwargs)
