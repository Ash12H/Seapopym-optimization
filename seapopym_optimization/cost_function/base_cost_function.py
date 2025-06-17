"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import numpy as np

    from seapopym_optimization.cost_function.base_observation import AbstractObservation
    from seapopym_optimization.model_generator.base_model_generator import AbstractModelGenerator


@dataclass(kw_only=True)
class AbstractCostFunction(ABC):
    """
    Abstract class for the cost function used in the optimization process.

    This class defines the interface for the cost function and provides a method to generate the partial cost function
    used for optimization. It requires a `FunctionalGroupSet`, a `ModelGenerator`, and a sequence of `Observations`.
    The `_cost_function` method must be implemented in the child class to calculate the cost of the simulation using the
    provided parameters (as a vector).
    """

    model_generator: AbstractModelGenerator
    observations: Sequence[AbstractObservation]
    functional_groups: FunctionalGroupSet

    def __post_init__(self: AbstractCostFunction) -> None:
        """Check types and convert functional groups if necessary."""
        if not isinstance(self.functional_groups, FunctionalGroupSet):
            self.functional_groups = FunctionalGroupSet(self.functional_groups)

        if not isinstance(self.observations, Sequence):
            msg = "Observations must be a Sequence of AbstractObservation."
            raise TypeError(msg)

    @abstractmethod
    def _cost_function(self: AbstractCostFunction, args: np.ndarray) -> tuple:
        """
        Calculate the cost of the simulation.

        This function must be rewritten in the child class.
        """

    def generate(self: AbstractCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(self._cost_function)
