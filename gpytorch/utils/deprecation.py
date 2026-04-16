#!/usr/bin/env python3

from __future__ import annotations

import functools
import warnings
from unittest.mock import MagicMock

import torch

# TODO: Use bool instead of uint8 dtype once pytorch #21113 is in stable release
if isinstance(torch, MagicMock):
    bool_compat = torch.uint8
else:
    bool_compat = (torch.ones(1) > 0).dtype


class DeprecationError(Exception):
    pass


def _deprecated_function_for(old_function_name, function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        pass

    return wrapper


def _deprecate_kwarg(kwargs, old_kw, new_kw, new_kw_value):
    pass


def _deprecate_kwarg_with_transform(kwargs, old_kw, new_kw, new_kw_value, transform):
    pass


def _deprecated_renamed_method(cls, old_method_name, new_method_name):
    def _deprecated_method(self, *args, **kwargs):
        pass

    _deprecated_method.__name__ = old_method_name
    setattr(cls, old_method_name, _deprecated_method)
    return cls


def _deprecate_renamed_methods(cls, **renamed_methods):
    pass
