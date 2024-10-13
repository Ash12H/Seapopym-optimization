"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
from dask.distributed import Client
from deap import algorithms, base, creator, tools

from seapopym_optimization.constraint import GenericConstraint

if TYPE_CHECKING:
    from seapopym_optimization.cost_function import GenericCostFunction, Parameter, GenericFunctionalGroupOptimize


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

        # NOTE(Jules): WITHOUT CREATOR -----------------------------

        class Fitness(base.Fitness):
            weights = self.cost_function_weight

        class Individual(list):
            def __init__(self: Individual, iterator):
                super().__init__(iterator)
                self.fitness = Fitness()

        for param in parameters:
            toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        def individual():
            return Individual([param.init_method(param.lower_bound, param.upper_bound) for param in parameters])

        toolbox.register("population", tools.initRepeat, list, individual)

        # ----------------------------------------------------------

        # NOTE(Jules): WITH CREATOR -----------------------------

        # creator.create("Fitness", base.Fitness, weights=self.cost_function_weight)
        # creator.create("Individual", list, fitness=creator.Fitness)
        # for param in parameters:
        #     toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        # toolbox.register(
        #     "individual",
        #     tools.initCycle,
        #     creator.Individual,
        #     tuple(toolbox.__dict__[param.name] for param in parameters),
        # )
        # toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # ----------------------------------------------------------

        toolbox.register("evaluate", cost_function.generate())
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
    parameter_genetic_algorithm: GeneticAlgorithmParameters
    cost_function: GenericCostFunction
    client: Client | None = None
    constraint: Sequence[GenericConstraint] | None = None

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        if self.client is None:
            self.client = Client()
        # TODO(Jules): Vérifier que les paramètres ont des noms uniques.

    @property
    def parameter_optimize(self: GeneticAlgorithm) -> Sequence[Parameter]:
        """The list of the parameters to optimize."""
        parameter_optimize = []
        for fg in self.cost_function.functional_groups:
            parameter_optimize += fg.get_parameters_to_optimize()
        return parameter_optimize

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
        # NOTE(Jules): Using individuals (i.e. Fitness) with Dask does not work. It must be converted to list.
        # But when converted to list, the constraint does not work.
        # invalid_ind_as_list = [list(ind) for ind in invalid_ind]  # For compatibility with DASK
        # futures_results = self.client.map(toolbox.evaluate, invalid_ind_as_list)
        #
        # NOTE(Jules): Here is the original version
        futures_results = self.client.map(toolbox.evaluate, invalid_ind)
        # -------------------------------------------------------------------------------------------------- #
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

            # NOTE(Jules): Using individuals (i.e. Fitness) with Dask does not work. It must be converted to list.
            # But when converted to list, the constraint does not work.
            # invalid_ind_as_list = [list(ind) for ind in invalid_ind]  # For compatibility with DASK
            # futures_results = self.client.map(toolbox.evaluate, invalid_ind_as_list)
            #
            # NOTE(Jules): Here is the original version
            futures_results = self.client.map(toolbox.evaluate, invalid_ind)
            # -------------------------------------------------------------------------------------------------- #

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

        # TODO(Jules): Pour le moment les contraintes sont incompatibles avec le calcul parallele avec DASK.
        # Le preoblème vient du fait que l'on utilise un list plutot que les structures définies par DEAP
        # CF. `invalid_ind_as_list`

        ordered_names = self.cost_function.parameters_name
        for constraint in self.constraint:
            toolbox.decorate("evaluate", constraint.generate(ordered_names))
        result = self.main(toolbox)
        return GeneticAlgorithmViewer(*result)
