from __future__ import annotations

from abc import ABC

import torch
from torch.nn import ModuleList

from gpytorch.likelihoods import LikelihoodList
from gpytorch.models import GP


class AbstractModelList(GP, ABC):
    def forward_i(self, i, *args, **kwargs):
        """Forward restricted to the i-th model only."""
        raise NotImplementedError

    def likelihood_i(self, i, *args, **kwargs):
        """Evaluate likelihood of the i-th model only."""
        raise NotImplementedError


class IndependentModelList(AbstractModelList):
    def __init__(self, *models):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "IndependentModelList currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])

    def forward_i(self, i, *args, **kwargs):
        pass

    def likelihood_i(self, i, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def get_fantasy_model(self, inputs, targets, **kwargs):
        """
        Returns a new GP model that incorporates the specified inputs and targets as new training data.

        This is a simple wrapper that creates fantasy models for each of the models in the model list,
        and returns the same class of fantasy models.

        Args:
            inputs: List of locations of fantasy observations, one for each model.
            targets List of labels of fantasy observations, one for each model.

        Returns:
            An `IndependentModelList` model, where each sub-model is the fantasy model of the respective
            sub-model in the original model at the corresponding input locations / labels.
        """
        pass

    def __call__(self, *args, **kwargs):
        return [
            model.__call__(*args_, **kwargs) for model, args_ in zip(self.models, _get_tensor_args(*args), strict=True)
        ]

    @property
    def train_inputs(self):
        pass

    @property
    def train_targets(self):
        pass


def _get_tensor_args(*args):
    pass
