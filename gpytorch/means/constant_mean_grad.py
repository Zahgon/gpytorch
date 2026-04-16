#!/usr/bin/env python3

from __future__ import annotations

import torch

from .mean import Mean


class ConstantMeanGrad(Mean):
    def __init__(self, prior=None, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constant", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, 1)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constant")

    def forward(self, input):
        pass
