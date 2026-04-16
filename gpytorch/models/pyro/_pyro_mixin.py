#!/usr/bin/env python3

from __future__ import annotations

import pyro
import torch


class _PyroMixin:
    def pyro_guide(self, input, beta=1.0, name_prefix=""):
        # Inducing values q(u)
        pass

    def pyro_model(self, input, beta=1.0, name_prefix=""):
        # Inducing values p(u)
        pass
