#!/usr/bin/env python3

from __future__ import annotations

import torch
from linear_operator import to_linear_operator
from linear_operator.operators import CatLinearOperator
from torch.nn.parallel import DataParallel

from .. import settings
from .kernel import Kernel


class MultiDeviceKernel(DataParallel, Kernel):
    r"""
    Allocates the covariance matrix on distributed devices, e.g. multiple GPUs.

    Args:
        base_kernel: Base kernel to distribute
        device_ids: list of `torch.device` objects to place kernel chunks on
        output_device: Device where outputs will be placed
    """

    def __init__(
        self,
        base_kernel: Kernel,
        device_ids: list[torch.device],
        output_device: torch.device | None = None,
        create_cuda_context: bool | None = True,
        **kwargs,
    ):
        # Need to warm up each GPU otherwise scattering in forward will be
        # EXTREMELY slow. This memory will be available as soon as we leave __init__
        if create_cuda_context:
            for d in device_ids:
                _ = torch.tensor([], device=d)

        DataParallel.__init__(self, module=base_kernel, device_ids=device_ids, output_device=output_device, dim=-2)

        self.output_device = output_device if output_device else device_ids[0]

        self.__cached_x1 = torch.empty(1)
        self.__cached_x2 = torch.empty(1)

    @property
    def base_kernel(self):
        pass

    def forward(self, x1, x2, diag=False, **kwargs):
        pass

    def gather(self, outputs, output_device):
        pass

    def num_outputs_per_input(self, x1, x2):
        pass
