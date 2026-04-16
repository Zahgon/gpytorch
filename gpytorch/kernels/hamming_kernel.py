from __future__ import annotations

import torch
from torch import nn, Tensor

from gpytorch.constraints.constraints import Interval, Positive
from gpytorch.kernels.kernel import Kernel
from gpytorch.priors.prior import Prior


EMPTY_SIZE = torch.Size([])


class HammingIMQKernel(Kernel):
    r"""
    Computes a covariance matrix based on the inverse multiquadratic Hamming kernel
    between inputs :math:`\mathbf{x_1}` and :math:`\mathbf{x_2}`:

    .. math::
       \begin{equation*}
            k_{\text{H-IMQ}}(\mathbf{x_1}, \mathbf{x_2}) =
            \left( \frac{1 + \alpha}{\alpha + d_{\text{Hamming}}(x1, x2)} \right)^\beta
       \end{equation*}
    where :math:`\alpha` and :math:`\beta` are strictly positive scale parameters.
    This kernel was proposed in `Biological Sequence Kernels with Guaranteed Flexibility`.
    See http://arxiv.org/abs/2304.03775 for more details.

    This kernel is meant to be used for fixed-length one-hot encoded discrete sequences.
    Because GPyTorch is particular about dimensions, the one-hot sequence encoding should be flattened
    to a vector with length :math:`T \times V`, where :math:`T` is the sequence length and :math:`V` is the
    vocabulary size.

    :param vocab_size: The size of the vocabulary.
    :param batch_shape: Set this if you want a separate kernel hyperparameters for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf{x_1}` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param alpha_prior: Set this if you want to apply a prior to the
        alpha parameter.
    :param: alpha_constraint: Set this if you want to apply a constraint
        to the alpha parameter. If None is passed, the default is `Positive()`.
    :param beta_prior: Set this if you want to apply a prior to the
        beta parameter.
    :param beta_constraint: Set this if you want to apply a constraint
        to the beta parameter. If None is passed, the default is `Positive()`.

    Example:
        >>> vocab_size = 8
        >>> x_cat = torch.tensor([[7, 7, 7, 7], [5, 7, 3, 4]])  # batch_size x seq_length
        >>> x_one_hot = F.one_hot(x_cat, num_classes=vocab_size)  # batch_size x seq_length x vocab_size
        >>> x_flat = x_one_hot.view(*x_cat.shape[:-1], -1)  # batch_size x (seq_length * vocab_size)
        >>> covar_module = gpytorch.kernels.HammingIMQKernel(vocab_size=vocab_size)
        >>> covar = covar_module(x_flat)  # Output: LinearOperator of size (2 x 2)
    """

    def __init__(
        self,
        vocab_size: int,
        batch_shape: torch.Size = EMPTY_SIZE,
        alpha_prior: Prior | None = None,
        alpha_constraint: Interval | None = None,
        beta_prior: Prior | None = None,
        beta_constraint: Interval | None = None,
    ):
        super().__init__(batch_shape=batch_shape)
        self.vocab_size = vocab_size
        # add alpha (scale) parameter
        alpha_constraint = Positive() if alpha_constraint is None else alpha_constraint
        self.register_parameter(
            name="raw_alpha",
            parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        if alpha_prior is not None:
            self.register_prior("alpha_prior", alpha_prior, self._alpha_param, self._alpha_closure)
        self.register_constraint("raw_alpha", alpha_constraint)

        # add beta parameter
        beta_constraint = Positive() if beta_constraint is None else beta_constraint
        self.register_parameter(
            name="raw_beta",
            parameter=nn.Parameter(torch.zeros(*self.batch_shape, 1)),
        )
        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, self._beta_param, self._beta_closure)
        self.register_constraint("raw_beta", beta_constraint)

    @property
    def alpha(self) -> Tensor:
        pass

    @alpha.setter
    def alpha(self, value: Tensor):
        pass

    def _alpha_param(self, m: Kernel) -> Tensor:
        # Used by the alpha_prior
        pass

    def _alpha_closure(self, m: Kernel, v: Tensor | float) -> None:
        # Used by the alpha_prior
        pass

    def _set_alpha(self, value: Tensor | float) -> None:
        # Used by the alpha_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    @property
    def beta(self) -> Tensor:
        pass

    @beta.setter
    def beta(self, value: Tensor):
        pass

    def _beta_param(self, m: Kernel) -> Tensor:
        # Used by the beta_prior
        pass

    def _beta_closure(self, m: Kernel, v: Tensor | float) -> None:
        # Used by the beta_prior
        pass

    def _set_beta(self, value: Tensor | float) -> None:
        # Used by the beta_prior
        if not isinstance(value, Tensor):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    def _imq(self, dist: Tensor) -> Tensor:
        pass

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params):
        # GPyTorch is pretty particular about dimensions so we need to unflatten the one-hot encoding
        pass


def hamming_dist(x1: Tensor, x2: Tensor, x1_eq_x2: bool) -> Tensor:
    pass
