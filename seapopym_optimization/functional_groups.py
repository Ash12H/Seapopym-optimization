"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from itertools import chain
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Callable, Sequence

import numpy as np

MAXIMUM_INIT_TRY = 1000


@dataclass
class Parameter:
    """
    The definition of a parameter to optimize.

    Parameters
    ----------
    name: str
        ...
    lower_bound: float
        ...
    upper_bound: float
        ...
    init_method: Callable[[float, float], float], optional
        The method used to get the initial value of a parameter. Default is a random uniform distribution that exclude
        the bounds values.

    """

    name: str
    lower_bound: float
    upper_bound: float
    init_method: Callable[[float, float], float] = None

    def __post_init__(self: Parameter) -> None:
        """Check that the parameter is correctly defined."""
        if self.lower_bound >= self.upper_bound:
            msg = f"Lower bounds ({self.lower_bound}) must be <= to upper bound ({self.upper_bound})."
            raise ValueError(msg)

        if self.init_method is None:

            def random_exclusive(lower: float, upper: float) -> float:
                count = 0
                while count < MAXIMUM_INIT_TRY:
                    value = random.uniform(lower, upper)
                    if value not in (lower, upper):
                        return value
                    count += 1
                msg = f"Random parameter initialization reach maximum try for parameter {self.name}"
                raise ValueError(msg)

            self.init_method = random_exclusive


@dataclass
class GenericFunctionalGroupOptimize(ABC):
    """The Generic structure used to store the parameters of a functional group as used in SeapoPym."""

    name: str

    @property
    def parameters(self: GenericFunctionalGroupOptimize) -> tuple:
        """
        Return the parameters representing the functional group.
        Order of declaration is the same as in the cost_function.
        """
        return tuple(getattr(self, field.name) for field in fields(self) if field.name not in excluded)

    def get_fixed_parameters(self: GenericFunctionalGroupOptimize, *, fill_with_name: float = True) -> tuple:
        """
        Return a tuple that contains all the functional group parameters (except name) as float values. When value is
        not set, return np.NAN.
        """
        return tuple(
            (param.name if fill_with_name else np.nan) if isinstance(param, Parameter) else param
            for param in self.parameters
        )

    def get_parameters_to_optimize(self: GenericFunctionalGroupOptimize) -> Sequence[Parameter]:
        """Return the parameters to optimize as a sequence of `Parameter`."""
        return tuple(param for param in self.parameters if isinstance(param, Parameter))


@dataclass
class FunctionalGroupOptimizeNoTransport(GenericFunctionalGroupOptimize):
    """The parameters of a functional group as they are defined in the SeapoPym NoTransport model."""

    day_layer: float | Parameter
    night_layer: float | Parameter
    energy_coefficient: float | Parameter
    tr_max: float | Parameter
    tr_rate: float | Parameter
    inv_lambda_max: float | Parameter
    inv_lambda_rate: float | Parameter


@dataclass
class AllGroups:
    functional_groups: Sequence[GenericFunctionalGroupOptimize]

    def get_all_names_ordered(self: AllGroups) -> Sequence[str]:
        all_param = tuple(chain.from_iterable(group.get_parameters_to_optimize() for group in self.functional_groups))
        all_names = tuple({param.name for param in all_param})
        # Remove duplicates and keep order
        return list(dict.fromkeys(all_names))

    def replace_strings_with_values(data_tuple, mapping_dict):
        return tuple(mapping_dict.get(item, item) if isinstance(item, str) else item for item in data_tuple)

    def generate_all_groups(self: AllGroups, x: Sequence[float]) -> np.ndarray:
        keys = self.get_all_names_ordered()
        parameters_values = dict(zip(keys, x))
        all_param = tuple(
            chain.from_iterable(group.get_fixed_parameters(fill_with_name=True) for group in self.functional_groups)
        )
        all_param = self.replace_strings_with_values(all_param, parameters_values)
        return np.array(all_param).reshape(len(self.functional_groups), -1)
