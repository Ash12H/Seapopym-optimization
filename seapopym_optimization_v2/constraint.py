"""All the constraints (as penalty functions) used by the DEAP library to contraint parameters initialization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

import numpy as np
from deap import tools

# TODO(Jules): Créer une structure qui permet de déterminer la position des paramètres à prendre en compte dans la
# contrainte par rapport aux individus.


def constraint_generator_no_transport_energy_coef(
    energy_coefficient_positions: Sequence[int],
    min_energy_coef_value: float,
    max_energy_coef_value: float,
) -> tools.DeltaPenalty:
    """Limit the total energy coefficient to a miximum of 100% (=1) in the NoTransport SeapoPym model."""

    def feasible(individual: Sequence[float]) -> bool:
        total_coef = sum([individual[i] for i in energy_coefficient_positions])
        return min_energy_coef_value <= total_coef <= max_energy_coef_value

    return tools.DeltaPenalty(feasibility=feasible, delta=np.inf)


@dataclass
class GenericConstraint(ABC):
    parameters_name: Sequence[str]

    def _generate_index(self: GenericConstraint, ordered_names: list[str]) -> list[int]:
        """
        List the index of the `parameters_name` in the `ordered_names` sequence. This should be used by the feasible
        function to retrive the position of the selected parameters.
        """
        return [ordered_names.index(param) for param in self.parameters_name]

    @abstractmethod
    def _feasible(self: GenericConstraint, selected_index: list[int]) -> Callable[[Sequence[float]], bool]:
        def feasible(individual: Sequence[float]) -> bool:
            """Rewrite this function."""

        return feasible

    def generate(self: GenericConstraint, ordered_names: list[str]) -> tools.DeltaPenalty:
        feasible = self._feasible(selected_index=self._generate_index(ordered_names))
        return tools.DeltaPenalty(feasibility=feasible, delta=np.inf)


@dataclass
class ConstraintNoTransportEnergyCoefficient(GenericConstraint):
    min_energy_coef_value: float
    max_energy_coef_value: float

    def _feasible(
        self: ConstraintNoTransportEnergyCoefficient, selected_index: list[int]
    ) -> Callable[[Sequence[float]], bool]:
        """The penalty when the sum of all energy transfert coefficients are greater than 1 or less than 0."""

        def feasible(individual: Sequence[float], min_coef: float, max_coef: float) -> bool:
            """Rewrite this function."""
            total_coef = sum([individual[i] for i in selected_index])
            return min_coef <= total_coef <= max_coef

        return partial(feasible, min_coef=self.min_energy_coef_value, max_coef=self.max_energy_coef_value)
