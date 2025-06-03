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
                    # TODO(Jules): There might be a better initialization method. Hypercube ?
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

    def as_dict(self: GenericFunctionalGroupOptimize) -> dict:
        """Return the attributes of the functional group as a dictionary."""
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name != "name"}

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
    """The structure used to generate the matrix of all parameters for all functional groups."""

    functional_groups: Sequence[GenericFunctionalGroupOptimize]

    @property
    def functional_groups_name(self: AllGroups) -> Sequence[str]:
        """Return the ordered list of the functional groups name."""
        return tuple(group.name for group in self.functional_groups)

    @property
    def unique_functional_groups_parameters_ordered(self: AllGroups) -> dict[str, Parameter]:
        """
        Return the unique optimized parameters of all functional groups in the order of declaration.

        Used to setup toolbox for optimization algorithms.
        """
        all_param = tuple(chain.from_iterable(group.get_parameters_to_optimize() for group in self.functional_groups))
        unique_params = {}
        for param in all_param:
            if param.name not in unique_params:
                unique_params[param.name] = param
        return unique_params

    # TODO(Jules): Rename, no more matrix
    def generate_matrix(self: AllGroups, x: Sequence[float]) -> list[dict[str, float]]:
        """
        Generate the matrix of all parameters for all functional groups. It can be used by the wrapper to generate the
        model.
        """

        def _replace_strings_with_values(data_tuple: tuple, mapping_dict: dict[str, float]) -> tuple:
            """Replace all strings in a tuple with their corresponding values in a dictionary."""
            return tuple(mapping_dict.get(item, item) if isinstance(item, str) else item for item in data_tuple)

        keys = self.unique_functional_groups_parameters_ordered.keys()
        all_param = tuple(
            chain.from_iterable(group.get_fixed_parameters(fill_with_name=True) for group in self.functional_groups)
        )
        try:
            parameters_values = dict(zip(keys, x, strict=True))
        except ValueError as e:
            msg = (
                f"Cost function parameters {x} do not match the expected parameters {keys}. "
                "Please check your parameters definition."
            )
            raise ValueError(msg) from e
        all_param = _replace_strings_with_values(all_param, parameters_values)
        all_param = np.array(all_param).reshape(len(self.functional_groups), -1)
        return [
            dict(zip(fgroup.as_dict().keys(), params_value))
            for fgroup, params_value in zip(self.functional_groups, all_param, strict=True)
        ]


if __name__ == "__main__":
    # Example usage
    fg1 = FunctionalGroupOptimizeNoTransport(
        name="FG1",
        day_layer=Parameter(name="day_layer", lower_bound=0, upper_bound=10),
        night_layer=1,
        energy_coefficient=Parameter(name="FG1_energy_coefficient", lower_bound=0, upper_bound=1),
        tr_max=Parameter(name="tr_max", lower_bound=0, upper_bound=100),
        tr_rate=Parameter(name="tr_rate", lower_bound=0, upper_bound=1),
        inv_lambda_max=Parameter(name="inv_lambda_max", lower_bound=0, upper_bound=10),
        inv_lambda_rate=Parameter(name="inv_lambda_rate", lower_bound=0, upper_bound=1),
    )

    fg2 = FunctionalGroupOptimizeNoTransport(
        name="FG2",
        day_layer=Parameter(name="day_layer", lower_bound=0, upper_bound=10),
        night_layer=2,
        energy_coefficient=Parameter(name="FG2_energy_coefficient", lower_bound=0, upper_bound=1),
        tr_max=Parameter(name="tr_max", lower_bound=0, upper_bound=100),
        tr_rate=Parameter(name="tr_rate", lower_bound=0, upper_bound=1),
        inv_lambda_max=Parameter(name="inv_lambda_max", lower_bound=0, upper_bound=10),
        inv_lambda_rate=Parameter(name="inv_lambda_rate", lower_bound=0, upper_bound=1),
    )

    all_groups = AllGroups(functional_groups=[fg1, fg2])
    print(all_groups.generate_matrix([5, 5, 50, 0.1, 5, 0.5, 66]))
