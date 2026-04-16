#!/usr/bin/env python3

from __future__ import annotations

import torch

from .mean import Mean


class ZeroMean(Mean):
    def __init__(self, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape

    def forward(self, input):
        pass
