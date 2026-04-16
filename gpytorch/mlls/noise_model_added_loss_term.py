#!/usr/bin/env python3

from __future__ import annotations

from .added_loss_term import AddedLossTerm


class NoiseModelAddedLossTerm(AddedLossTerm):
    def __init__(self, noise_model):
        from .exact_marginal_log_likelihood import ExactMarginalLogLikelihood

        self.noise_mll = ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)

    def loss(self, *params):
        pass
