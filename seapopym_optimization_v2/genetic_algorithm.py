"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
from deap import algorithms, base, tools
from IPython.display import clear_output, display
from tqdm import tqdm

from seapopym_optimization.viewer import GeneticAlgorithmViewer, _compute_stats

if TYPE_CHECKING:
    from dask.distributed import Client

    from seapopym_optimization.constraint import GenericConstraint
    from seapopym_optimization.cost_function import GenericCostFunction
    from seapopym_optimization.functional_groups import Parameter


@dataclass
class GeneticAlgorithmParameters:
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
    """
    Crowding degree of the mutation. A high eta will produce a mutant resembling its parent, while a small eta will
    produce a solution much more different.
    """
    INDPB: float
    """Represents the individual probability of mutation for each attribute of the individual."""
    CXPB: float
    """Represents the probability of mating two individuals."""
    MUTPB: float
    """Represents the probability of mutating an individual."""
    NGEN: int
    """Represents the number of generations."""
    POP_SIZE: int
    """Represents the size of the population."""
    cost_function_weight: tuple | float = (-1.0,)
    """The weight of the cost function. The default value is (-1.0,) to minimize the cost function."""

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
        - mate
        - mutate
        - select
        """
        toolbox = base.Toolbox()

        # NOTE(Jules): WITHOUT CREATOR -----------------------------
        # I use self made structure because Dask doesn't work with the individuals created with deap.creator.

        class Fitness(base.Fitness):
            weights = self.cost_function_weight

        class Individual(list):
            def __init__(self: Individual, iterator):
                super().__init__(iterator)
                self.fitness = Fitness()

        toolbox.register("Individual", Individual)

        for param in parameters:
            toolbox.register(param.name, param.init_method, param.lower_bound, param.upper_bound)

        def individual():
            return Individual([param.init_method(param.lower_bound, param.upper_bound) for param in parameters])

        toolbox.register("population", tools.initRepeat, list, individual)
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
        # NOTE(Jules): La nouvelle population peut contenir des individus en plusieurs exemplaires là où d'autres
        # individus peuvent être absents.
        return toolbox


# TODO(Jules): Use Param library rather than dataclass ?
@dataclass
class GeneticAlgorithm:
    """
    Contains the genetic algorithm parameters and thA quoi correspondent e cost function to optimize. By default, the order of
    of the process is SCM: Select, Cross, Mutate.
    """

    parameter_genetic_algorithm: GeneticAlgorithmParameters
    cost_function: GenericCostFunction
    client: Client | None = None
    constraint: Sequence[GenericConstraint] | None = None
    logbook_path: Path | str | None = None

    logbook: tools.Logbook | None = field(default=None, init=False, repr=False)
    toolbox: base.Toolbox | None = field(default=None, init=False, repr=False)

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        # TODO(Jules): Vérifier que les paramètres ont des noms uniques.
        # LOGBOOK
        if self.logbook_path is not None:
            if not isinstance(self.logbook_path, Path):
                self.logbook_path = Path(self.logbook_path)
            if self.logbook_path.exists():
                self.logbook = pd.read_json(self.logbook_path, orient="table")
        # TOOLBOX
        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered
        self.toolbox = self.parameter_genetic_algorithm.generate_toolbox(
            ordered_parameters.values(), self.cost_function
        )
        if self.constraint is not None:
            for constraint in self.constraint:
                self.toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))

    def update_logbook(self: GeneticAlgorithm, logbook: tools.Logbook) -> None:
        """Update the logbook."""
        if self.logbook is None:
            self.logbook = logbook
        else:
            self.logbook = pd.concat([self.logbook, logbook])

    def _helper_core_manage_inf(self: GeneticAlgorithm, func: Callable, *args, **kwargs) -> Callable:
        """Transforme the function to manage np.inf values returned by the constraints."""

        def wrapper(arg):
            arg = np.asarray(arg)
            valide = np.isfinite(arg)
            if np.sum(valide) == 0:
                return np.nan
            arg = arg[valide]
            return func(arg, *args, **kwargs)

        return wrapper

    def _helper_core_evaluate(
        self: GeneticAlgorithm,
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
            individuals, columns=self.cost_function.functional_groups.unique_functional_groups_parameters_ordered.keys()
        )
        scores = [ind.fitness.values for ind in individuals]
        final_scores = np.dot(scores, self.parameter_genetic_algorithm.cost_function_weight)
        df_logbook["fitness"] = scores
        df_logbook["fitness_final"] = final_scores
        df_logbook["previous_generation"] = known
        df_logbook["generation"] = generation
        df_logbook.index.name = "individual"
        df_logbook = df_logbook.reset_index()
        return df_logbook.set_index(["generation", "previous_generation", "individual"]).sort_index()

    def _helper_core_initialization(self: GeneticAlgorithm) -> tuple[int, list[base.Individual]]:
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
            for individual, individual_fitness in zip(individuals_values, population_unprocessed["fitness"]):
                indiv = self.toolbox.Individual(individual)
                indiv.fitness.values = individual_fitness
                population.append(indiv)
            return last_computed_generation + 1, population
        return 0, []

    def _core(self: GeneticAlgorithm) -> None:
        """
        The core function as it is described in the DEAP documentation. It is adapted to allow Dask client for
        parallel computing. The order used is SCM: Select, Cross, Mutate.
        """
        # INIT POPULATION IF LOGBOOK NOT EMPTY
        first_generation, population = self._helper_core_initialization()

        # RUN THE GENETIC ALGORITHM
        for gen in tqdm(desc="Generations", iterable=range(first_generation, self.parameter_genetic_algorithm.NGEN)):
            if gen == 0:
                population = self.toolbox.population(n=self.parameter_genetic_algorithm.POP_SIZE)
                df_logbook = self._helper_core_evaluate(self.toolbox, population, gen)
            else:
                offspring = self.toolbox.select(population, len(population))
                offspring = algorithms.varAnd(
                    offspring,
                    self.toolbox,
                    self.parameter_genetic_algorithm.CXPB,
                    self.parameter_genetic_algorithm.MUTPB,
                )  # MUTATE + MATE
                df_logbook = self._helper_core_evaluate(self.toolbox, offspring, gen)
                population[:] = offspring

                # TODO(Jules): L'intégralité de la population est remplacée par les nouveaux individus. Je pourrai ajouter
                # paramètre GGAP pour conserver les meilleurs individus de la génération précédente.
                # Cf. Maria Angelova and Tania Pencheva 2011 - Tuning Genetic Algorithm Parameters to Improve Convergence
                # Time

            self.update_logbook(df_logbook)
            clear_output(wait=True)
            display(_compute_stats(self.logbook))
            if self.logbook_path is not None:
                self.logbook.to_json(self.logbook_path, orient="table")

    def optimize(self: GeneticAlgorithm) -> GeneticAlgorithmViewer:
        """This is the main function. Use it to optimize your model."""
        self._core()
        return GeneticAlgorithmViewer(
            parameters=self.cost_function.functional_groups,
            forcing_parameters=self.cost_function.forcing_parameters,
            logbook=self.logbook.copy(),
            observations=self.cost_function.observations,
        )
