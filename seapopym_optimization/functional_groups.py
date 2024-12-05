"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

import random
from abc import ABC
from dataclasses import dataclass, fields
from itertools import chain
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
        excluded = ("name",)
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

    # NOTE(Jules): Be sure that you respect the order of the parameters as defined in the wrapper module.
    day_layer: float | Parameter
    night_layer: float | Parameter
    energy_coefficient: float | Parameter
    tr_max: float | Parameter
    tr_rate: float | Parameter
    inv_lambda_max: float | Parameter
    inv_lambda_rate: float | Parameter


@dataclass
class AllGroups:
    """The structure used to generate the matrix of all parameters for all functional groups."""

    functional_groups: Sequence[GenericFunctionalGroupOptimize]

    @property
    def functional_groups_name(self: AllGroups) -> Sequence[str]:
        """Return the ordered list of the functional groups name."""
        return tuple(group.name for group in self.functional_groups)

    @property
    def unique_functional_groups_parameters_ordered(self: AllGroups) -> dict[str, Parameter]:
        """Return the unique parameters of all functional groups in the order of declaration."""
        all_param = tuple(chain.from_iterable(group.get_parameters_to_optimize() for group in self.functional_groups))
        unique_params = {}
        for param in all_param:
            if param.name not in unique_params:
                unique_params[param.name] = param
        return unique_params

    def _replace_strings_with_values(self: AllGroups, data_tuple: tuple, mapping_dict: dict[str, float]) -> tuple:
        """Replace all strings in a tuple with their corresponding values in a dictionary."""
        return tuple(mapping_dict.get(item, item) if isinstance(item, str) else item for item in data_tuple)

    def generate_matrix(self: AllGroups, x: Sequence[float]) -> np.ndarray:
        """Generate the matrix of all parameters for all functional groups. This can be used to generate the model."""
        keys = self.unique_functional_groups_parameters_ordered.keys()
        parameters_values = dict(zip(keys, x))
        all_param = tuple(
            chain.from_iterable(group.get_fixed_parameters(fill_with_name=True) for group in self.functional_groups)
        )
        all_param = self._replace_strings_with_values(all_param, parameters_values)
        return np.array(all_param).reshape(len(self.functional_groups), -1)
