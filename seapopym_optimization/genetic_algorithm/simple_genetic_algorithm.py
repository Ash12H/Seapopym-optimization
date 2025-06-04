"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from deap import algorithms, base, tools
from IPython.display import clear_output, display
from tqdm import tqdm

from seapopym_optimization.genetic_algorithm.base_genetic_algorithm import (
    AbstractGeneticAlgorithmParameters,
    individual_creator,
)
from seapopym_optimization.viewer.viewer import GeneticAlgorithmViewer, _compute_stats

if TYPE_CHECKING:
    from collections.abc import Sequence
    from os import PathLike

    from dask.distributed import Client

    from seapopym_optimization.constraint.energy_transfert_constraint import GenericConstraint
    from seapopym_optimization.cost_function import GenericCostFunction
    from seapopym_optimization.functional_group.no_transport_functional_groups import Parameter


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

    mate: callable = field(default=tools.cxTwoPoint, init=False)

    ETA: float
    INDPB: float
    CXPB: float
    MUTPB: float
    NGEN: int
    POP_SIZE: int
    TOURNSIZE: int = field(default=3)
    cost_function_weight: tuple = (-1.0,)

    def __post_init__(self: SimpleGeneticAlgorithmParameters) -> None:
        self.select = tools.selTournament
        self.mate = tools.cxTwoPoint
        self.mutate = tools.mutPolynomialBounded
        self.variation = algorithms.varAnd

    def generate_toolbox(
        self: SimpleGeneticAlgorithmParameters, parameters: Sequence[Parameter], cost_function: GenericCostFunction
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
    cost_function: GenericCostFunction
        The cost function to optimize.
    client: Client | None
        The Dask client to use for parallel computing. If None, the algorithm will run in serial.
    constraint: Sequence[GenericConstraint] | None
        The constraints to apply to the individuals. If None, no constraints are applied.
    logbook_path: PathLike | None
        The path to the logbook file. If None, the logbook is not saved to a file. The logbook is a json file generated
        with pandas and can be read with `pd.read_json(logbook_path, orient="table")`.

    """

    meta_parameter: SimpleGeneticAlgorithmParameters
    cost_function: GenericCostFunction
    client: Client | None = None
    constraint: Sequence[GenericConstraint] | None = None
    logbook_path: PathLike | None = None

    logbook: tools.Logbook | None = field(default=None, init=False, repr=False)
    toolbox: base.Toolbox | None = field(default=None, init=False, repr=False)

    def __post_init__(self: SimpleGeneticAlgorithm) -> None:
        """Check parameters."""
        if self.logbook_path is not None:
            if not isinstance(self.logbook_path, Path):
                self.logbook_path = Path(self.logbook_path)
            if self.logbook_path.exists():
                self.logbook = pd.read_json(self.logbook_path, orient="table")

        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered()
        self.toolbox = self.meta_parameter.generate_toolbox(ordered_parameters.values(), self.cost_function)
        if self.constraint is not None:
            for constraint in self.constraint:
                self.toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))

    def _initialization(self: SimpleGeneticAlgorithm) -> tuple[int, list[base.Individual]]:
        """
        Initialize the population.

        Returns
        -------
        int
            The first generation number (0 if the `logbook` is empty).
        list[base.Individual]
            The initialized population (an empty list if the `logbook`is empty).

        """
        if self.logbook is not None:
            population_unprocessed = self.logbook.reset_index()
            last_computed_generation = population_unprocessed["generation"].max()
            population_unprocessed = population_unprocessed.query("generation == @last_computed_generation")

            individuals_values = population_unprocessed.drop(
                columns=["generation", "individual", "previous_generation", "fitness", "fitness_final"]
            ).to_numpy()
            population = []
            for individual, individual_fitness in zip(
                individuals_values, population_unprocessed["fitness"], strict=True
            ):
                indiv = self.toolbox.Individual(individual)
                indiv.fitness.values = individual_fitness  # TODO(Jules): Si valeur nulle alors calculer le fitness
                population.append(indiv)
            return last_computed_generation + 1, population
        return 0, []

    def _evaluate(
        self: SimpleGeneticAlgorithm,
        toolbox: base.Toolbox,
        individuals: Sequence,
        generation: int,
    ) -> tuple[tools.Logbook, tools.HallOfFame]:
        """Evaluate the cost function of all new individuals and update the statistiques."""
        # 1 - UPDATE POPULATION
        known = [ind.fitness.valid for ind in individuals]
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        if self.client is None:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
        else:
            futures_results = self.client.map(toolbox.evaluate, invalid_ind)
            fitnesses = self.client.gather(futures_results)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 2 - GENERATE LOGBOOK
        df_logbook = pd.DataFrame(
            individuals,
            columns=self.cost_function.functional_groups.unique_functional_groups_parameters_ordered().keys(),
        )
        scores = [ind.fitness.values for ind in individuals]
        final_scores = np.dot(scores, self.meta_parameter.cost_function_weight)
        df_logbook["fitness"] = scores
        df_logbook["fitness_final"] = final_scores
        df_logbook["previous_generation"] = known
        df_logbook["generation"] = generation
        df_logbook.index.name = "individual"
        df_logbook = df_logbook.reset_index()
        return df_logbook.set_index(["generation", "previous_generation", "individual"]).sort_index()

    def optimize(self: SimpleGeneticAlgorithm) -> GeneticAlgorithmViewer:
        """This is the main function. Use it to optimize your model."""

        def update_logbook(logbook: tools.Logbook) -> None:
            """Update the logbook."""
            if self.logbook is None:
                self.logbook = logbook
            else:
                self.logbook = pd.concat([self.logbook, logbook])

        first_generation, population = self._initialization()

        for gen in tqdm(desc="Generations", iterable=range(first_generation, self.meta_parameter.NGEN)):
            if gen == 0:
                population = self.toolbox.population(n=self.meta_parameter.POP_SIZE)
                df_logbook = self._evaluate(self.toolbox, population, gen)
            else:
                offspring = self.toolbox.select(population, len(population))
                offspring = self.meta_parameter.variation(
                    offspring, self.toolbox, self.meta_parameter.CXPB, self.meta_parameter.MUTPB
                )
                df_logbook = self._evaluate(self.toolbox, offspring, gen)
                population[:] = offspring

            update_logbook(df_logbook)
            clear_output(wait=True)
            display(_compute_stats(self.logbook))
            if self.logbook_path is not None:
                self.logbook.to_json(self.logbook_path, orient="table")

        return GeneticAlgorithmViewer(
            parameters=self.cost_function.functional_groups,
            forcing_parameters=self.cost_function.forcing_parameters,
            logbook=self.logbook.copy(),
            observations=self.cost_function.observations,
        )
