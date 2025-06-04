"""All the constraints (as penalty functions) used by the DEAP library to contraint parameters initialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import numpy as np
from deap import tools


@dataclass
class AbstractConstraint(ABC):
    """
    Abstract class for defining constraints used by the DEAP library to apply penalties on individuals that do not
    satisfy the constraints.
    This class should be inherited to define specific constraints.

    Attributes
    ----------
        parameters_name: Sequence[str]
            The names of the parameters that are involved in the constraint.

    """

    parameters_name: Sequence[str]

    @abstractmethod
    def _feasible(self: AbstractConstraint, selected_index: list[int]) -> Callable[[Sequence[float]], bool]:
        """
        Define the feasibility function used by the genetic algorithm to apply the penalty if the constraint is not
        satisfied.
        """

        def feasible(individual: Sequence[float]) -> bool:  # noqa: ARG001
            """Rewrite this function."""

        return partial(feasible)

    def generate(self: AbstractConstraint, ordered_names: list[str]) -> tools.DeltaPenalty:
        """
        Generate the DeltaPenalty object used by the DEAP library to apply the penalty on individuals that do not
        satisfy the constraint.
        """

        def generate_index(ordered_names: list[str]) -> list[int]:
            """
            List the index of the `parameters_name` in the `ordered_names` sequence. This should be used by the feasible
            function to retrive the position of the selected parameters.
            """
            return [ordered_names.index(param) for param in self.parameters_name]

        feasible = self._feasible(selected_index=generate_index(ordered_names))
        return tools.DeltaPenalty(feasibility=feasible, delta=np.inf)
