from __future__ import annotations

from torch.nn import ModuleList

from gpytorch.likelihoods import Likelihood


def _get_tuple_args_(*args):
    pass


class LikelihoodList(Likelihood):
    def __init__(self, *likelihoods):
        super().__init__()
        self.likelihoods = ModuleList(likelihoods)

    def expected_log_prob(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        pass

    def pyro_sample_output(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if "noise" in kwargs:
            noise = kwargs.pop("noise")
            # if noise kwarg is passed, assume it's an iterable of noise tensors
            return [
                likelihood(*args_, {**kwargs, "noise": noise_})
                for likelihood, args_, noise_ in zip(self.likelihoods, _get_tuple_args_(*args), noise, strict=True)
            ]
        else:
            return [
                likelihood(*args_, **kwargs)
                for likelihood, args_ in zip(self.likelihoods, _get_tuple_args_(*args), strict=True)
            ]
