"""
A module that contains the base class for functional groups declaration and parameter management in optimization
process.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, fields
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np

from seapopym_optimization.functional_group.parameter_initialization import random_uniform_exclusive

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


@dataclass
class Parameter:
    """
    The definition of a parameter to optimize.

    Parameters
    ----------
    name: str
        The name of the parameter.
    lower_bound: float
        The lower bound of the parameter.
    upper_bound: float
        The upper bound of the parameter.
    init_method: Callable[[float, float], float], optional
        The method used to get the initial value of a parameter. Default is a random uniform distribution that exclude
        the bounds values.

    """

    name: str
    lower_bound: float
    upper_bound: float
    init_method: Callable[[float, float], float] = random_uniform_exclusive

    def __post_init__(self: Parameter) -> None:
        """Check that the parameter is correctly defined."""
        if self.lower_bound >= self.upper_bound:
            msg = f"Lower bounds ({self.lower_bound}) must be <= to upper bound ({self.upper_bound})."
            raise ValueError(msg)


@dataclass
class AbstractFunctionalGroup(ABC):
    """The Generic structure used to store the parameters of a functional group as used in SeapoPym."""

    name: str

    @property
    def parameters(self: AbstractFunctionalGroup) -> tuple:
        """Return the parameters representing the functional group. Order of declaration is preserved."""
        excluded = ("name",)
        return tuple(getattr(self, field.name) for field in fields(self) if field.name not in excluded)

    def as_dict(self: AbstractFunctionalGroup) -> dict:
        """Return the functional group as a dictionary with parameter names as keys (without functional group name)."""
        return {field.name: getattr(self, field.name) for field in fields(self) if field.name != "name"}

    def get_fixed_parameters(self: AbstractFunctionalGroup, *, fill_with_name: float = True) -> tuple:
        """
        Return a tuple that contains all the functional group parameters (except name) as float values. When value is
        not set, return np.NAN.
        """
        return tuple(
            (param.name if fill_with_name else np.nan) if isinstance(param, Parameter) else param
            for param in self.parameters
        )

    def get_parameters_to_optimize(self: AbstractFunctionalGroup) -> Sequence[Parameter]:
        """Return the parameters to optimize as a sequence of `Parameter`."""
        return tuple(param for param in self.parameters if isinstance(param, Parameter))


@dataclass
class FunctionalGroupSet:
    """The structure used to generate the matrix of all parameters for all functional groups."""

    functional_groups: Sequence[AbstractFunctionalGroup]

    def functional_groups_name(self: FunctionalGroupSet) -> Sequence[str]:
        """Return the ordered list of the functional groups name."""
        return tuple(group.name for group in self.functional_groups)

    def unique_functional_groups_parameters_ordered(self: FunctionalGroupSet) -> dict[str, Parameter]:
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

    def generate(self: FunctionalGroupSet, x: Sequence[float]) -> list[dict[str, float]]:
        """
        Generate a list of dictionaries representing the functional groups with their parameters values.
        The order of the parameters is defined by the `unique_functional_groups_parameters_ordered` method.
        The input `x` should match the order of the parameters returned by that method.
        It is used by the `model_generator` to generate the model.

        Parameters
        ----------
        x: Sequence[float]
            A sequence of float values representing the parameters to set for each functional group.

        Returns
        -------
        list[dict[str, float]]
            A list of dictionaries where each dictionary represents a functional group with its parameters and their
            corresponding values.

        """
        keys = list(self.unique_functional_groups_parameters_ordered().keys())

        try:
            parameters_values = dict(zip(keys, x, strict=True))
        except ValueError as e:
            msg = (
                f"Cost function parameters {x} do not match the expected parameters {keys}. "
                "Please check your parameters definition."
            )
            raise ValueError(msg) from e

        result = []
        for group in self.functional_groups:
            param_names = list(group.as_dict().keys())
            param_values = [
                parameters_values.get(param.name, np.nan) if isinstance(param, Parameter) else param
                for param in group.parameters
            ]
            result.append(dict(zip(param_names, param_values, strict=True)))
        return result

    # NOTE(Jules): Old version of the `generate` method. Kept for reference but not used. Should be removed if
    # everything works as expected with the new version.

    # def _replace_strings_with_values(data_tuple: tuple, mapping_dict: dict[str, float]) -> tuple:
    #     """Replace all strings in a tuple with their corresponding values in a dictionary."""
    #     return tuple(mapping_dict.get(item, item) if isinstance(item, str) else item for item in data_tuple)

    # keys = self.unique_functional_groups_parameters_ordered().keys()
    # all_param = tuple(
    #     chain.from_iterable(group.get_fixed_parameters(fill_with_name=True) for group in self.functional_groups)
    # )
    # try:
    #     parameters_values = dict(zip(keys, x, strict=True))
    # except ValueError as e:
    #     msg = (
    #         f"Cost function parameters {x} do not match the expected parameters {keys}. "
    #         "Please check your parameters definition."
    #     )
    #     raise ValueError(msg) from e
    # all_param = _replace_strings_with_values(all_param, parameters_values)
    # all_param = np.array(all_param).reshape(len(self.functional_groups), -1)
    # return [
    #     dict(zip(fgroup.as_dict().keys(), params_value))
    #     for fgroup, params_value in zip(self.functional_groups, all_param, strict=True)
    # ]
