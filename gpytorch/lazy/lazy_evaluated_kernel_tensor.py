#!/usr/bin/env python3

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from linear_operator import LinearOperator, to_linear_operator
from linear_operator.utils.getitem import _noop_index

from .. import beta_features, settings
from ..utils import deprecation
from ..utils.memoize import cached


def recall_grad_state(method: Callable) -> Callable:
    """Decorator for LazyEvaluatedKernelTensor's methods to put their execution
    inside the same grad state as during the instantiation of the lazy object.
    This makes the lazy tensor object behave in the same way as a regular tensor
    with respect to the grad state.
    """
    pass


class LazyEvaluatedKernelTensor(LinearOperator):
    _check_size = False

    def _check_args(self, x1, x2, kernel, last_dim_is_batch=False, **params):
        pass

    def __init__(self, x1, x2, kernel, last_dim_is_batch=False, **params):
        super().__init__(x1, x2, kernel=kernel, last_dim_is_batch=last_dim_is_batch, **params)
        self.kernel = kernel
        self.x1 = x1
        self.x2 = x2
        self.last_dim_is_batch = last_dim_is_batch
        self.params = params
        self._is_grad_enabled = torch.is_grad_enabled()  # records grad state at instantiation

    @property
    def dtype(self):
        pass

    @property
    def device(self):
        return self.x1.device

    @property
    def requires_grad(self):
        pass

    def _set_requires_grad(self, val):
        pass

    def _bilinear_derivative(self, left_vecs, right_vecs):
        # This _bilinear_derivative computes the kernel in chunks
        # It is only used when we are using kernel checkpointing
        # It won't be called if checkpointing is off
        pass

    @cached(name="kernel_diag")
    @recall_grad_state
    def _diagonal(self) -> torch.Tensor:
        # Getting the diagonal of a kernel can be handled more efficiently by
        # transposing the batch and data dimension before calling the kernel.
        # Implementing it this way allows us to compute predictions more efficiently
        # in cases where only the variances are required.
        pass

    @recall_grad_state
    def _getitem(self, row_index, col_index, *batch_indices):
        pass

    def _matmul(self, rhs):
        # This _matmul is defined computes the kernel in chunks
        # It is only used when we are using kernel checkpointing
        # It won't be called if checkpointing is off
        pass

    @cached(name="size")
    def _size(self):
        pass

    @recall_grad_state
    def _transpose_nonbatch(self):
        pass

    @recall_grad_state
    def _unsqueeze_batch(self, dim):
        pass

    @cached(name="kernel_eval")
    @recall_grad_state
    def evaluate_kernel(self):
        """
        NB: This is a meta LinearOperator, in the sense that evaluate can return
        a LinearOperator if the kernel being evaluated does so.
        """
        x1 = self.x1
        x2 = self.x2

        with settings.lazily_evaluate_kernels(False):
            temp_active_dims = self.kernel.active_dims
            self.kernel.active_dims = None
            res = self.kernel(
                x1,
                x2,
                diag=False,
                last_dim_is_batch=self.last_dim_is_batch,
                **self.params,
            )
            self.kernel.active_dims = temp_active_dims

        # Check the size of the output
        if settings.debug.on():
            if res.shape != self.shape:
                raise RuntimeError(
                    f"The expected shape of the kernel was {self.shape}, but got {res.shape}. "
                    "This is likely a bug in GPyTorch."
                )

        return to_linear_operator(res)

    @recall_grad_state
    def repeat(self, *repeats):
        if len(repeats) == 1 and hasattr(repeats[0], "__iter__"):
            repeats = repeats[0]
        *batch_repeat, row_repeat, col_repeat = repeats

        x1 = self.x1.repeat(*batch_repeat, row_repeat, 1)
        x2 = self.x2.repeat(*batch_repeat, col_repeat, 1)
        return self.__class__(
            x1,
            x2,
            kernel=self.kernel,
            last_dim_is_batch=self.last_dim_is_batch,
            **self.params,
        )

    def representation(self):
        # If we're checkpointing the kernel, we'll use chunked _matmuls defined in LazyEvaluatedKernelTensor
        pass

    def representation_tree(self):
        # If we're checkpointing the kernel, we'll use chunked _matmuls defined in LazyEvaluatedKernelTensor
        pass

    @cached
    def to_dense(self):
        return self.evaluate_kernel().to_dense()

    @recall_grad_state
    def __getitem__(self, index):
        """
        Supports subindexing of the matrix this LinearOperator represents. This may return either another
        :obj:`~linear_operator.operators.LinearOperator` or a :obj:`torch.tensor` depending on the exact implementation.
        """
        # Process the index
        index = index if isinstance(index, tuple) else (index,)
        # Special case for the most common case: [..., slice, slice]
        if len(index) == 3 and index[0] is Ellipsis and isinstance(index[1], slice) and isinstance(index[2], slice):
            _, row_index, col_index = index
            batch_indices = [slice(None, None, None)] * (self.dim() - 2)
            return self._getitem(row_index, col_index, *batch_indices)
        else:
            return super().__getitem__(index)


deprecation._deprecated_renamed_method(
    LazyEvaluatedKernelTensor, old_method_name="_quad_form_derivative", new_method_name="_bilinear_derivative"
)
deprecation._deprecated_renamed_method(LazyEvaluatedKernelTensor, old_method_name="diag", new_method_name="diagonal")
deprecation._deprecated_renamed_method(
    LazyEvaluatedKernelTensor, old_method_name="evaluate", new_method_name="to_dense"
)
