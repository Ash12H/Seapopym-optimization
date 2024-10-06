"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

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


# TODO(Jules): Use Param library rather than dataclass ?
@dataclass
class GeneticAlgorithm:
    parameters: Sequence[Parameter]
    ETA: float
    INDPB: float
    CXPB: float
    MUTPB: float
    NGEN: int
    POP_SIZE: int
    cost_function_weight: tuple | float = (1.0,)
    hall_of_fame_size: int

    def __post_init__(self: GeneticAlgorithm) -> None:
        pass
