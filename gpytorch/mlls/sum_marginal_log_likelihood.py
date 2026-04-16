from __future__ import annotations

from torch.nn import ModuleList

from gpytorch.mlls import ExactMarginalLogLikelihood, MarginalLogLikelihood


class SumMarginalLogLikelihood(MarginalLogLikelihood):
    """Sum of marginal log likelihoods, to be used with Multi-Output models.

    Args:
        likelihood: A MultiOutputLikelihood
        model: A MultiOutputModel
        mll_cls: The Marginal Log Likelihood class (default: ExactMarginalLogLikelihood)

    In case the model outputs are independent, this provives the MLL of the multi-output model.

    """

    def __init__(self, likelihood, model, mll_cls=ExactMarginalLogLikelihood):
        super().__init__(model.likelihood, model)
        self.mlls = ModuleList([mll_cls(mdl.likelihood, mdl) for mdl in model.models])

    def forward(self, outputs, targets, *params):
        """
        Args:
            outputs: (Iterable[MultivariateNormal]) - the outputs of the latent function
            targets: (Iterable[Tensor]) - the target values
            params: (Iterable[Iterable[Tensor]]) - the arguments to be passed through
                (e.g. parameters in case of heteroskedastic likelihoods)
        """
        pass
