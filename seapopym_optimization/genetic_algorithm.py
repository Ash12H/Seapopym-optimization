"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
from dask.distributed import Client
from deap import algorithms, base, creator, tools

if TYPE_CHECKING:
    from seapopym_optimization.cost_function import GenericCostFunction

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


@dataclass
class GeneticAlgorithmParameters:
    """
    The structure used to store the genetic algorithm parameters. Can generate the toolbox with default
    parameters.
    """

    ETA: float
    INDPB: float
    CXPB: float
    MUTPB: float
    NGEN: int
    POP_SIZE: int
    cost_function_weight: tuple | float = (-1.0,)
    hall_of_fame_size: int = 100

    # TODO(Jules): Add default parameters for : mate method, mutate method, select method

    def __post_init__(self: GeneticAlgorithmParameters) -> None:
        """Check parameters."""
        if sum(self.cost_function_weight) != 1:
            msg = "The sum of the cost function weight must equal 1."
            raise ValueError(msg)

    def generate_toolbox(
        self: GeneticAlgorithmParameters, parameters: Sequence[Parameter], cost_function: GenericCostFunction
    ) -> base.Toolbox:
        """
        Default behaviour of the toolbox. Any definition can be rewrite after the toolbox generation. Just use the
        toolbox.register() method to overwrite.

        Parameters in toolbox :
        -----------------------

        - Fitness
        - Individual
        - individual
        - population
        - evaluate
        - amte
        - mutate
        - select
        """
        toolbox = base.Toolbox()

        creator.create("Fitness", base.Fitness, weights=self.cost_function_weight)
        creator.create("Individual", list, fitness=creator.Fitness)

        for param in parameters:
            toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            tuple(toolbox.__dict__[param.name] for param in parameters),
            n=1,
        )

        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", cost_function)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register(
            "mutate",
            tools.mutPolynomialBounded,
            eta=self.ETA,
            indpb=self.INDPB,
            low=[param.lower_bound for param in parameters],
            up=[param.upper_bound for param in parameters],
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        return toolbox


@dataclass
class GeneticAlgorithmViewer:
    """
    Structure that contains the output of the optimization. Use the representation to plot some informations about the
    results.
    """

    population: Sequence[Sequence[float]]
    logbook: tools.Logbook
    hall_of_fame: tools.HallOfFame

    def show(self: GeneticAlgorithmViewer):
        """Show informations about the optimization results in different plots."""


# TODO(Jules): Use Param library rather than dataclass ?
@dataclass
class GeneticAlgorithm:
    parameter_optimize: Sequence[Parameter]
    parameter_genetic_algorithm: GeneticAlgorithmParameters
    cost_function: GenericCostFunction
    client: Client = None

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        if self.client is None:
            self.client = Client()

    def main(
        self: GeneticAlgorithm, toolbox: base.Toolbox
    ) -> tuple[Sequence[Sequence[float]], tools.Logbook, tools.HallOfFame]:
        """
        The main function as it is desrcibed in the DEAP documentation. It is adapted to include Dask client for
        parallel computing.
        """
        population = toolbox.population(n=self.parameter_genetic_algorithm.POP_SIZE)
        halloffame = tools.HallOfFame(self.parameter_genetic_algorithm.hall_of_fame_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        invalid_ind_as_list = [list(ind) for ind in invalid_ind]  # For compatibility with DASK
        futures_results = self.client.map(toolbox.evaluate, invalid_ind_as_list)
        fitnesses = self.client.gather(futures_results)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)

        for gen in range(1, self.parameter_genetic_algorithm.NGEN + 1):
            offspring = toolbox.select(population, len(population))

            offspring = algorithms.varAnd(
                offspring, toolbox, self.parameter_genetic_algorithm.CXPB, self.parameter_genetic_algorithm.MUTPB
            )  # MUTATE + MATE

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            invalid_ind_as_list = [list(ind) for ind in invalid_ind]  # For compatibility with DASK
            futures_results = self.client.map(toolbox.evaluate, invalid_ind_as_list)
            fitnesses = self.client.gather(futures_results)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            if halloffame is not None:
                halloffame.update(offspring)

            population[:] = offspring

            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

        return population, logbook, halloffame

    def optimize(self: GeneticAlgorithm) -> GeneticAlgorithmViewer:
        toolbox = self.parameter_genetic_algorithm.generate_toolbox(self.parameter_optimize, self.cost_function)
        result = self.main(toolbox)
        return GeneticAlgorithmViewer(result)
