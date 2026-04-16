#!/usr/bin/env python3

from __future__ import annotations

import warnings
from abc import abstractmethod
from collections.abc import Callable, Iterable
from copy import deepcopy

import torch
from linear_operator import to_dense, to_linear_operator
from linear_operator.operators import LinearOperator, ZeroLinearOperator
from torch import Tensor
from torch.nn import ModuleList

from .. import settings
from ..constraints import Interval, Positive
from ..distributions import MultivariateNormal
from ..lazy import LazyEvaluatedKernelTensor
from ..likelihoods import GaussianLikelihood
from ..models import exact_prediction_strategies
from ..module import Module
from ..priors import Prior


def sq_dist(x1, x2, x1_eq_x2=False):
    """Equivalent to the square of `torch.cdist` with p=2."""
    pass


def dist(x1, x2, x1_eq_x2=False):
    """
    Equivalent to `torch.cdist` with p=2, but clamps the minimum element to 1e-15.
    """
    pass


# only necessary for legacy purposes
class Distance(torch.nn.Module):
    def __init__(self, postprocess: Callable | None = None):
        super().__init__()
        if postprocess is not None:
            warnings.warn(
                "The `postprocess` argument is deprecated. "
                "See https://github.com/cornellius-gp/gpytorch/pull/2205 for details.",
                DeprecationWarning,
            )
        self._postprocess = postprocess

    def _sq_dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        pass

    def _dist(self, x1, x2, x1_eq_x2=False, postprocess=False):
        pass


class Kernel(Module):
    r"""
    Kernels in GPyTorch are implemented as a :class:`gpytorch.Module` that, when called on two :class:`torch.Tensor`
    objects :math:`\mathbf x_1` and :math:`\mathbf x_2` returns either a :obj:`torch.Tensor` or a
    :obj:`~linear_operator.operators.LinearOperator` that represents the
    covariance matrix between :math:`\mathbf x_1` and :math:`\mathbf x_2`.

    In the typical use case, extend this class simply requires implementing a
    :py:meth:`~gpytorch.kernels.Kernel.forward` method.

    .. note::
        The :py:meth:`~gpytorch.kernels.Kernel.__call__` method does some additional internal work. In particular,
        all kernels are lazily evaluated so that we can index in to the kernel matrix before actually
        computing it. Furthermore, many built-in kernel modules return
        :class:`~linear_operator.LinearOperators` that allow for more efficient
        inference than if we explicitly computed the kernel matrix itself.

        As a result, if you want to get an actual
        :obj:`torch.tensor` representing the covariance matrix, you may need to call the
        :func:`~linear_operator.operators.LinearOperator.to_dense` method on the output.

    This base :class:`Kernel` class includes a lengthscale parameter
    :math:`\Theta`, which is used by many common kernel functions.
    There are a few options for the lengthscale:

    * Default: No lengthscale (i.e. :math:`\Theta` is the identity matrix).

    * Single lengthscale: One lengthscale can be applied to all input dimensions/batches
      (i.e. :math:`\Theta` is a constant diagonal matrix).
      This is controlled by setting the attribute `has_lengthscale=True`.

    * ARD: Each input dimension gets its own separate lengthscale
      (i.e. :math:`\Theta` is a non-constant diagonal matrix).
      This is controlled by the `ard_num_dims` keyword argument (as well as `has_lengthscale=True`).

    In batch mode (i.e. when :math:`\mathbf x_1` and :math:`\mathbf x_2` are batches of input matrices), each
    batch of data can have its own lengthscale parameter by setting the `batch_shape`
    keyword argument to the appropriate number of batches.

    .. note::

        You can set a prior on the lengthscale parameter using the lengthscale_prior argument.

    :param ard_num_dims: Set this if you want a separate lengthscale for each input
        dimension. It should be `D` if :math:`\mathbf x` is a `... x N x D` matrix. (Default: `None`.)
    :param batch_shape: Set this if you want a separate lengthscale for each batch of input
        data. It should be :math:`B_1 \times \ldots \times B_k` if :math:`\mathbf x_1` is
        a :math:`B_1 \times \ldots \times B_k \times N \times D` tensor.
    :param active_dims: Set this if you want to compute the covariance of only
        a few input dimensions. The ints corresponds to the indices of the
        dimensions. (Default: `None`.)
    :param lengthscale_prior: Set this if you want to apply a prior to the
        lengthscale parameter. (Default: `None`.)
    :param lengthscale_constraint: Set this if you want to apply a constraint
        to the lengthscale parameter. (Default: :class:`~gpytorch.constraints.Positive`.)

    :ivar torch.Size batch_shape:
        The (minimum) number of batch dimensions supported by this kernel.
        Typically, this captures the batch shape of the lengthscale and other parameters,
        and is usually set by the `batch_shape` argument in the constructor.
    :ivar torch.dtype dtype:
        The dtype supported by this kernel.
        Typically, this depends on the dtype of the lengthscale and other parameters.
    :ivar bool is_stationary:
        Set to True if the Kernel represents a stationary function
        (one that depends only on :math:`\mathbf x_1 - \mathbf x_2`).
    :ivar torch.Tensor lengthscale:
        The lengthscale parameter. Size/shape of parameter depends on the
        `ard_num_dims` and `batch_shape` arguments.

    Example:
        >>> covar_module = gpytorch.kernels.LinearKernel()
        >>> x1 = torch.randn(50, 3)
        >>> lazy_covar_matrix = covar_module(x1) # Returns a RootLinearOperator
        >>> tensor_covar_matrix = lazy_covar_matrix.to_dense() # Gets the actual tensor for this kernel matrix
    """

    has_lengthscale = False

    def __init__(
        self,
        ard_num_dims: int | None = None,
        batch_shape: torch.Size | None = None,
        active_dims: tuple[int, ...] | None = None,
        lengthscale_prior: Prior | None = None,
        lengthscale_constraint: Interval | None = None,
        **kwargs,
    ):
        super().__init__()
        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.register_buffer("active_dims", active_dims)
        self.ard_num_dims = ard_num_dims

        param_transform = kwargs.get("param_transform")

        if lengthscale_constraint is None:
            lengthscale_constraint = Positive()

        if param_transform is not None:
            warnings.warn(
                "The 'param_transform' argument is now deprecated. If you want to use a different "
                "transformation, specify a different 'lengthscale_constraint' instead.",
                DeprecationWarning,
            )

        if self.has_lengthscale:
            lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
            self.register_parameter(
                name="raw_lengthscale",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, lengthscale_num_dims)),
            )
            if lengthscale_prior is not None:
                if not isinstance(lengthscale_prior, Prior):
                    raise TypeError("Expected gpytorch.priors.Prior but got " + type(lengthscale_prior).__name__)
                self.register_prior(
                    "lengthscale_prior", lengthscale_prior, self._lengthscale_param, self._lengthscale_closure
                )

            self.register_constraint("raw_lengthscale", lengthscale_constraint)

        self.distance_module = None
        # TODO: Remove this on next official PyTorch release.
        self.__pdist_supports_batch = True

    def _lengthscale_param(self, m: Kernel) -> Tensor:
        # Used by the lengthscale_prior
        pass

    def _lengthscale_closure(self, m: Kernel, v: Tensor) -> None:
        # Used by the lengthscale_prior
        pass

    def _set_lengthscale(self, value: Tensor):
        # Used by the lengthscale_prior
        pass

    @abstractmethod
    def forward(
        self, x1: Tensor, x2: Tensor, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> Tensor | LinearOperator:
        r"""
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.
        This method should be implemented by all Kernel subclasses.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)

        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        raise NotImplementedError()

    @property
    def batch_shape(self) -> torch.Size:
        pass

    @batch_shape.setter
    def batch_shape(self, val: torch.Size):
        pass

    @property
    def device(self) -> torch.device | None:
        if self.has_lengthscale:
            return self.lengthscale.device
        devices = {param.device for param in self.parameters()}
        if len(devices) > 1:
            raise RuntimeError(f"The kernel's parameters are on multiple devices: {devices}.")
        elif devices:
            return devices.pop()
        return None

    @property
    def dtype(self) -> torch.dtype:
        pass

    @property
    def lengthscale(self) -> Tensor:
        pass

    @lengthscale.setter
    def lengthscale(self, value: Tensor):
        pass

    @property
    def is_stationary(self) -> bool:
        pass

    def local_load_samples(self, samples_dict: dict[str, Tensor], memo: set, prefix: str):
        pass

    def covar_dist(
        self,
        x1: Tensor,
        x2: Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        square_dist: bool = False,
        **params,
    ) -> Tensor:
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)
        :param square_dist:
            If True, returns the squared distance rather than the standard distance. (Default: False.)
        :return: The kernel matrix or vector. The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        pass

    def expand_batch(self, *sizes: torch.Size | tuple[int, ...]) -> Kernel:
        r"""
        Constructs a new kernel where the lengthscale (and other kernel parameters)
        are expanded to match the batch dimension determined by `sizes`.

        :param sizes: The batch shape of the new tensor
        """
        pass

    def named_sub_kernels(self) -> Iterable[tuple[str, Kernel]]:
        """
        For compositional Kernel classes (e.g. :class:`~gpytorch.kernels.AdditiveKernel`
        or :class:`~gpytorch.kernels.ProductKernel`).

        :return: An iterator over the component kernel objects,
            along with the name of each component kernel.
        """
        pass

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        """
        For most kernels, `num_outputs_per_input = 1`.

        However, some kernels (e.g. multitask kernels or interdomain kernels) return a
        `num_outputs_per_input x num_outputs_per_input` matrix of covariance values for
        every pair of data points.

        I.e. if `x1` is size `... x N x D` and `x2` is size `... x M x D`, then the size of the kernel
        will be `... x (N * num_outputs_per_input) x (M * num_outputs_per_input)`.

        :return: `num_outputs_per_input` (usually 1).
        """
        pass

    def prediction_strategy(
        self,
        train_inputs: Tensor,
        train_prior_dist: MultivariateNormal,
        train_labels: Tensor,
        likelihood: GaussianLikelihood,
    ) -> exact_prediction_strategies.PredictionStrategy:
        return exact_prediction_strategies.DefaultPredictionStrategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )

    def sub_kernels(self) -> Iterable[Kernel]:
        """
        For compositional Kernel classes (e.g. :class:`~gpytorch.kernels.AdditiveKernel`
        or :class:`~gpytorch.kernels.ProductKernel`).

        :return: An iterator over the component kernel objects.
        """
        pass

    def __call__(
        self, x1: Tensor, x2: Tensor | None = None, diag: bool = False, last_dim_is_batch: bool = False, **params
    ) -> LazyEvaluatedKernelTensor | LinearOperator | Tensor:
        r"""
        Computes the covariance between :math:`\mathbf x_1` and :math:`\mathbf x_2`.

        .. note::
            Following PyTorch convention, all :class:`~gpytorch.models.GP` objects should use `__call__`
            rather than :py:meth:`~gpytorch.kernels.Kernel.forward`.
            The `__call__` method applies additional pre- and post-processing to the `forward` method,
            and additionally employs a lazy evaluation scheme to reduce memory and computational costs.

        :param x1: First set of data (... x N x D).
        :param x2: Second set of data (... x M x D).
            (If `None`, then `x2` is set to `x1`.)
        :param diag: Should the Kernel compute the whole kernel, or just the diag?
            If True, it must be the case that `x1 == x2`. (Default: False.)
        :param last_dim_is_batch: If True, treat the last dimension
            of `x1` and `x2` as another batch dimension.
            (Useful for additive structure over the dimensions). (Default: False.)

        :return: An object that will lazily evaluate to the kernel matrix or vector.
            The shape depends on the kernel's evaluation mode:

            * `full_covar`: `... x N x M`
            * `full_covar` with `last_dim_is_batch=True`: `... x K x N x M`
            * `diag`: `... x N`
            * `diag` with `last_dim_is_batch=True`: `... x K x N`
        """
        if last_dim_is_batch:
            warnings.warn(
                "The last_dim_is_batch argument is deprecated, and will be removed in GPyTorch 2.0. "
                "If you are using it as part of AdditiveStructureKernel or ProductStructureKernel, "
                'please update your code according to the "Kernels with Additive or Product Structure" '
                "tutorial in the GPyTorch docs.",
                DeprecationWarning,
            )

        x1_, x2_ = x1, x2

        # Select the active dimensions
        if self.active_dims is not None:
            x1_ = x1_.index_select(-1, self.active_dims)
            if x2_ is not None:
                x2_ = x2_.index_select(-1, self.active_dims)

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        # Check that ard_num_dims matches the supplied number of dimensions
        if settings.debug.on():
            if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
                raise RuntimeError(
                    "Expected the input to have {} dimensionality "
                    "(based on the ard_num_dims argument). Got {}.".format(self.ard_num_dims, x1_.size(-1))
                )

        if diag:
            res = super().__call__(x1_, x2_, diag=True, last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diagonal(dim1=-1, dim2=-2)
            return res

        else:
            if settings.lazily_evaluate_kernels.on():
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = to_linear_operator(super().__call__(x1_, x2_, last_dim_is_batch=last_dim_is_batch, **params))
            return res

    def __getstate__(self):
        # JIT ScriptModules cannot be pickled
        self.distance_module = None
        return self.__dict__

    def __add__(self, other: Kernel) -> Kernel:
        kernels = []
        kernels += self.kernels if isinstance(self, AdditiveKernel) else [self]
        kernels += other.kernels if isinstance(other, AdditiveKernel) else [other]
        return AdditiveKernel(*kernels)

    def __mul__(self, other: Kernel) -> Kernel:
        kernels = []
        kernels += self.kernels if isinstance(self, ProductKernel) else [self]
        kernels += other.kernels if isinstance(other, ProductKernel) else [other]
        return ProductKernel(*kernels)

    def __setstate__(self, d):
        self.__dict__ = d

    def __getitem__(self, index) -> Kernel:
        r"""
        Constructs a new kernel where the lengthscale (and other kernel parameters)
        are modified by an indexing operation.

        :param index: Index to apply to all parameters.
        """

        if len(self.batch_shape) == 0:
            return self

        new_kernel = deepcopy(self)
        # Process the index
        index = index if isinstance(index, tuple) else (index,)

        for param_name, param in self.named_parameters(recurse=False):
            new_param = new_kernel.__getattr__(param_name)
            new_param.data = new_param.__getitem__(index)
            ndim_removed = len(param.shape) - len(new_param.shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_param.shape[:new_batch_shape_len]

        for buffr_name, buffr in self.named_buffers(recurse=False):
            # For a given buffer, get the number of dimensions that do not correspond to the batch shape
            new_buffr = new_kernel.__getattr__(buffr_name)
            new_buffr.data = new_buffr.__getitem__(index)
            ndim_removed = len(buffr.shape) - len(new_buffr.shape)
            new_batch_shape_len = len(self.batch_shape) - ndim_removed
            new_kernel.batch_shape = new_buffr.shape[:new_batch_shape_len]

        for sub_module_name, sub_module in self.named_sub_kernels():
            new_kernel.__setattr__(sub_module_name, sub_module.__getitem__(index))

        return new_kernel


class AdditiveKernel(Kernel):
    """
    A Kernel that supports summing over multiple component kernels.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) + RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> additive_kernel_matrix = covar_module(x1)

    :param kernels: Kernels to add together.
    """

    @property
    def is_stationary(self) -> bool:
        pass

    def __init__(self, *kernels: Kernel):
        super().__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor | LinearOperator:
        pass

    def num_outputs_per_input(self, x1, x2):
        pass

    def __getitem__(self, index) -> Kernel:
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = kernel.__getitem__(index)

        return new_kernel


class ProductKernel(Kernel):
    """
    A Kernel that supports elementwise multiplying multiple component kernels together.

    Example:
        >>> covar_module = RBFKernel(active_dims=torch.tensor([1])) * RBFKernel(active_dims=torch.tensor([2]))
        >>> x1 = torch.randn(50, 2)
        >>> kernel_matrix = covar_module(x1) # The RBF Kernel already decomposes multiplicatively, so this is foolish!

    :param kernels: Kernels to multiply together.
    """

    @property
    def is_stationary(self) -> bool:
        pass

    def __init__(self, *kernels: Kernel):
        super().__init__()
        self.kernels = ModuleList(kernels)

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor | LinearOperator:
        pass

    def num_outputs_per_input(self, x1: Tensor, x2: Tensor) -> int:
        pass

    def __getitem__(self, index) -> Kernel:
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = kernel.__getitem__(index)

        return new_kernel
