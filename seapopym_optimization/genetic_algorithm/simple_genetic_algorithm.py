"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from deap import algorithms, base, tools

from seapopym_optimization.genetic_algorithm.base_genetic_algorithm import (
    AbstractGeneticAlgorithmParameters,
    individual_creator,
)
from seapopym_optimization.genetic_algorithm.simple_logbook import Logbook, LogbookCategory, LogbookIndex
from seapopym_optimization.viewer.simple_viewer import SimpleViewer

if TYPE_CHECKING:
    from collections.abc import Sequence
    from numbers import Number

    import pandas as pd
    from dask.distributed import Client
    from pandas._typing import FilePath, WriteBuffer

    from seapopym_optimization.constraint.energy_transfert_constraint import AbstractConstraint
    from seapopym_optimization.cost_function.base_cost_function import AbstractCostFunction
    from seapopym_optimization.functional_group.no_transport_functional_groups import Parameter

logger = logging.getLogger(__name__)


@dataclass
class SimpleGeneticAlgorithmParameters(AbstractGeneticAlgorithmParameters):
    """
    The structure used to store the genetic algorithm parameters. Can generate the toolbox with default
    parameters.

    Parameters
    ----------
    MUTPB: float
        Represents the probability of mutating an individual. It is recommended to use a value between 0.001 and 0.1.
    ETA: float
        Crowding degree of the mutation. A high eta will produce a mutant resembling its parent, while a small eta will
        produce a solution much more different. It is recommended to use a value between 1 and 20.
    INDPB: float
        Represents the individual probability of mutation for each attribute of the individual. It is recommended to use
        a value between 0.0 and 0.1. If you have a lot of parameters, you can use a 1/len(parameters) value.
    CXPB: float
        Represents the probability of mating two individuals. It is recommended to use a value between 0.5 and 1.0.
    NGEN: int
        Represents the number of generations.
    POP_SIZE: int
        Represents the size of the population.
    cost_function_weight: tuple | float = (-1.0,)
        The weight of the cost function. The default value is (-1.0,) to minimize the cost function.

    """

    ETA: float
    INDPB: float
    CXPB: float
    MUTPB: float
    NGEN: int
    POP_SIZE: int
    TOURNSIZE: int = field(default=3)
    cost_function_weight: tuple[Number] = (-1.0,)

    def __post_init__(self: SimpleGeneticAlgorithmParameters) -> None:
        self.select = tools.selTournament
        self.mate = tools.cxTwoPoint
        self.mutate = tools.mutPolynomialBounded
        self.variation = algorithms.varAnd
        self.cost_function_weight = tuple(
            np.asarray(self.cost_function_weight) / np.sum(np.absolute(self.cost_function_weight))
        )

    def generate_toolbox(
        self: SimpleGeneticAlgorithmParameters, parameters: Sequence[Parameter], cost_function: AbstractCostFunction
    ) -> base.Toolbox:
        """Generate a DEAP toolbox with the necessary functions for the genetic algorithm."""
        toolbox = base.Toolbox()
        Individual = individual_creator(self.cost_function_weight)  # noqa: N806
        toolbox.register("Individual", Individual)

        for param in parameters:
            toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        def individual() -> list:
            return Individual([param.init_method(param.lower_bound, param.upper_bound) for param in parameters])

        toolbox.register("population", tools.initRepeat, list, individual)
        toolbox.register("evaluate", cost_function.generate())
        toolbox.register("mate", self.mate)
        low_boundaries = [param.lower_bound for param in parameters]
        up_boundaries = [param.upper_bound for param in parameters]
        toolbox.register("mutate", self.mutate, eta=self.ETA, indpb=self.INDPB, low=low_boundaries, up=up_boundaries)
        toolbox.register("select", self.select, tournsize=self.TOURNSIZE)
        return toolbox


@dataclass
class SimpleGeneticAlgorithm:
    """
    Contains the genetic algorithm parameters and the cost function to optimize. By default, the order of
    of the process is SCM: Select, Cross, Mutate.

    Attributes
    ----------
    meta_parameter: SimpleGeneticAlgorithmParameters
        The parameters of the genetic algorithm.
    cost_function: AbstractCostFunction
        The cost function to optimize.
    client: Client | None
        The Dask client to use for parallel computing. If None, the algorithm will run in serial.
    constraint: Sequence[AbstractConstraint] | None
        The constraints to apply to the individuals. If None, no constraints are applied.
    save: PathLike | None
        The path to save the logbook (in parquet format). If None, the logbook is not saved.

    """

    meta_parameter: SimpleGeneticAlgorithmParameters
    cost_function: AbstractCostFunction
    client: Client | None = None
    constraint: Sequence[AbstractConstraint] | None = None

    save: FilePath | WriteBuffer[bytes] | None = None
    logbook: Logbook | None = field(default=None, repr=False)
    toolbox: base.Toolbox | None = field(default=None, init=False, repr=False)

    def __post_init__(self: SimpleGeneticAlgorithm) -> None:
        """Check parameters."""
        if self.save is not None:
            self.save = Path(self.save)
            if self.save.exists():
                waring_msg = f"Logbook file {self.save} already exists. Please choose a different path."
                logger.warning(waring_msg)

        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered()
        self.toolbox = self.meta_parameter.generate_toolbox(ordered_parameters.values(), self.cost_function)

        if self.constraint is not None:
            for constraint in self.constraint:
                self.toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))

    def update_logbook(self: SimpleGeneticAlgorithm, logbook: Logbook) -> None:
        """Update the logbook with the new data and save to disk if a path is provided."""
        if not isinstance(logbook, Logbook):
            logbook = Logbook(logbook)

        self.logbook = logbook if self.logbook is None else self.logbook.append_new_generation(logbook)

        if self.save is not None:
            self.logbook.to_parquet(self.save)

    def _evaluate(self: SimpleGeneticAlgorithm, individuals: Sequence, generation: int) -> Logbook:
        """Evaluate the cost function of all new individuals and update the statistiques."""

        def update_fitness(individuals: list) -> list:
            known = [ind.fitness.valid for ind in individuals]
            invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
            if self.client is None:
                fitnesses = list(map(self.toolbox.evaluate, invalid_ind))
            else:
                futures_results = self.client.map(self.toolbox.evaluate, invalid_ind)
                fitnesses = self.client.gather(futures_results)
            for ind, fit in zip(invalid_ind, fitnesses, strict=True):
                ind.fitness.values = fit
            return known

        known = update_fitness(individuals)

        return Logbook.from_individual(
            generation=generation,
            is_from_previous_generation=known,
            individual=individuals,
            parameter_names=self.cost_function.functional_groups.unique_functional_groups_parameters_ordered().keys(),
            fitness_name=[obs.name for obs in self.cost_function.observations],
        )

    def _initialization(self: SimpleGeneticAlgorithm) -> tuple[int, list[list]]:
        """Initialize the genetic algorithm. If a logbook is provided, it will load the last generation."""

        def create_first_generation() -> tuple[Literal[1], list[list]]:
            """Create the first generation (i.e. generation `0`) of individuals."""
            new_generation = 0
            population = self.toolbox.population(n=self.meta_parameter.POP_SIZE)
            logbook = self._evaluate(individuals=population, generation=new_generation)
            self.update_logbook(logbook)
            next_generation = new_generation + 1
            return next_generation, population

        def create_population_from_logbook(population_unprocessed: pd.DataFrame) -> list[list]:
            """Create a population from the logbook DataFrame."""
            individuals = population_unprocessed.loc[:, [LogbookCategory.PARAMETER]].to_numpy()
            fitness = list(population_unprocessed.loc[:, [LogbookCategory.FITNESS]].itertuples(index=False, name=None))
            fitness = [() if any(np.isnan(fit)) else fit for fit in fitness]
            return [
                self.toolbox.Individual(iterator=iterator, values=values)
                for iterator, values in zip(individuals, fitness, strict=True)
            ]

        if self.logbook is None:
            return create_first_generation()

        logger.info("Logbook found. Loading last generation.")

        last_computed_generation = self.logbook.index.get_level_values(LogbookIndex.GENERATION).max()
        population_unprocessed = self.logbook.loc[last_computed_generation]

        population = create_population_from_logbook(population_unprocessed)

        if population_unprocessed.loc[:, LogbookCategory.FITNESS].isna().any(axis=None):
            logger.warning("Some individuals in the logbook have no fitness values. Re-evaluating the population.")
            logbook = self._evaluate(population, last_computed_generation)
            self.logbook = None
            self.update_logbook(logbook)

        return last_computed_generation + 1, population

    def optimize(self: SimpleGeneticAlgorithm) -> None:  # -> GeneticAlgorithmViewer:
        """This is the main function. Use it to optimize your model."""
        generation_start, population = self._initialization()

        for gen in range(generation_start, self.meta_parameter.NGEN):
            offspring = self.toolbox.select(population, self.meta_parameter.POP_SIZE)
            offspring = self.meta_parameter.variation(
                offspring, self.toolbox, self.meta_parameter.CXPB, self.meta_parameter.MUTPB
            )
            logbook = self._evaluate(offspring, gen)

            self.update_logbook(logbook)
            population[:] = offspring

        return SimpleViewer(
            logbook=self.logbook.copy(),
            functional_group_set=self.cost_function.functional_groups,
            model_generator=self.cost_function.model_generator,
            observations=self.cost_function.observations,
            cost_function_weight=self.meta_parameter.cost_function_weight,
        )
