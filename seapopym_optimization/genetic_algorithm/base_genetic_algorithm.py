"""Base classes for genetic algorithms in SeapoPym optimization."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from deap import base

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    from seapopym_optimization.constraint.base_constraint import AbstractConstraint
    from seapopym_optimization.cost_function.base_cost_function import AbstractCostFunction


def individual_creator(cost_function_weight: tuple[Number]) -> type:
    """
    Create a custom individual class for DEAP genetic algorithms.

    This individual class inherits from `list` and includes a fitness attribute. It is redefined to work with the
    Dask framework, which does not support the default DEAP individual structure created with `deap.creator.create`.
    """

    class Fitness(base.Fitness):
        """Fitness class to store the fitness of an individual."""

        weights = cost_function_weight

    class Individual(list):
        """Individual class to store the parameters of an individual."""

        def __init__(self: Individual, iterator: Sequence, values: Sequence[Number] = ()) -> None:
            super().__init__(iterator)
            self.fitness = Fitness(values=values)

    return Individual


@dataclass
class AbstractGeneticAlgorithmParameters(ABC):
    """Base class for parameters of a genetic algorithm."""

    @abstractmethod
    def generate_toolbox(self: AbstractGeneticAlgorithmParameters) -> base.Toolbox:
        """Return a DEAP toolbox configured with the necessary genetic algorithm functions."""


@dataclass
class AbstractGeneticAlgorithm(ABC):
    """Base class for a genetic algorithm implementation."""

    parameter: AbstractGeneticAlgorithmParameters
    cost_function: AbstractCostFunction
    constraint: Sequence[AbstractConstraint] = field(default_factory=list)

    @abstractmethod
    def optimize() -> AbstractViewer:
        """Run the optimization algorithm and return a structure containing the results."""
