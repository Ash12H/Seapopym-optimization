"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
from dask.distributed import Client
from deap import algorithms, base, tools
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from seapopym_optimization.constraint import GenericConstraint
    from seapopym_optimization.cost_function import GenericCostFunction, Parameter


@dataclass
class GeneticAlgorithmParameters:
    """
    The structure used to store the genetic algorithm parameters. Can generate the toolbox with default
    parameters.

    # TODO(Jules): Describe the parameters

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
        # I use self made structure because Dask doesn't work with the individuals created with deap.creator.

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

    _parameters: Sequence[Parameter]
    _population: Sequence[Sequence[float]]
    _logbook: tools.Logbook
    _hall_of_fame: tools.HallOfFame

    @property
    def parameters_names(self: GeneticAlgorithmViewer):
        return [fg.name for fg in self._parameters]

    @property
    def parameters_lower_bounds(self: GeneticAlgorithmViewer):
        return [fg.lower_bound for fg in self._parameters]

    @property
    def parameters_upper_bound(self: GeneticAlgorithmViewer):
        return [fg.upper_bound for fg in self._parameters]

    @property
    def logbook(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """A review of the generations stats."""
        return pd.DataFrame(self._logbook)

    @property
    def hall_of_fame(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """The best individuals and their fitness."""
        hof = pd.DataFrame(self._hall_of_fame, columns=self.parameters_names)
        hof["fitness"] = [ind.fitness.values[0] for ind in self._hall_of_fame]
        return hof[np.isfinite(hof["fitness"])]

    def box_plot(self: GeneticAlgorithmViewer, columns_number: int):
        nb_fig = len(self.parameters_names)
        nb_row = nb_fig // columns_number + (1 if nb_fig % columns_number > 0 else 0)

        fig = make_subplots(rows=nb_row, cols=columns_number, subplot_titles=self.parameters_names)

        for i, (a, b, c) in enumerate(
            zip(self.parameters_names, self.parameters_lower_bounds, self.parameters_upper_bound)
        ):
            fig.add_trace(
                px.box(data_frame=self.hall_of_fame, y=a, range_y=(b, c), title=a).data[0],
                row=(i // columns_number) + 1,
                col=(i % columns_number) + 1,
            )

        return fig

    def parallel_coordinates(self: GeneticAlgorithmViewer):
        """TODO(Jules): Let the user select the number of individual to print."""
        hof_fitness = self.hall_of_fame
        hof_fitness["fitness"] = hof_fitness["fitness"] / hof_fitness["fitness"].max()
        fig = px.parallel_coordinates(
            hof_fitness,
            color="fitness",
            dimensions=self.parameters_names,
            labels=self.parameters_names,
            color_continuous_scale=[[0, "rgb(255,0,0,0.5)"], [0.6, "rgb(200,0,0,0.5)"], [1, "green"]],
            title="BATS parameters optimization",
        )

        fig.update_layout(coloraxis_colorbar={"title": "Fitness"})
        return fig


# TODO(Jules): Use Param library rather than dataclass ?
@dataclass
class GeneticAlgorithm:
    parameter_genetic_algorithm: GeneticAlgorithmParameters
    cost_function: GenericCostFunction
    client: Client | None = None
    constraint: Sequence[GenericConstraint] | None = None

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        # TODO(Jules): Vérifier que les paramètres ont des noms uniques.

    @property
    def parameter_optimize(self: GeneticAlgorithm) -> Sequence[Parameter]:
        """The list of the parameters to optimize."""
        parameter_optimize = []
        for fg in self.cost_function.functional_groups:
            parameter_optimize += fg.get_parameters_to_optimize()
        return parameter_optimize

    def _helper_core_manage_inf(self: GeneticAlgorithm, func: Callable) -> Callable:
        """Transforme the function to manage np.inf values returned by the constraints."""

        def wrapper(arg):
            arg = np.asarray(arg)
            valide = np.isfinite(arg)
            if np.sum(valide) == 0:
                return np.nan
            arg = arg[valide]
            return func(arg)

        return wrapper

    def _helper_core_initialize_viewer_containt(
        self: GeneticAlgorithm,
    ) -> tuple[tools.HallOfFame, tools.Logbook, tools.Statistics]:
        """Initialize all the structures needed by the `viewer`."""
        halloffame = tools.HallOfFame(self.parameter_genetic_algorithm.hall_of_fame_size)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", self._helper_core_manage_inf(np.nanmean))
        stats.register("std", self._helper_core_manage_inf(np.nanstd))
        stats.register("min", self._helper_core_manage_inf(np.nanmin))
        stats.register("max", self._helper_core_manage_inf(np.nanmax))
        stats.register("nvalide", lambda x: np.isfinite(x).sum())
        stats.register("ninvalide", lambda x: (~np.isfinite(x)).sum())

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        return halloffame, logbook, stats

    def _helper_core_evaluate(
        self: GeneticAlgorithm,
        toolbox: base.Toolbox,
        individuals: Sequence,
        logbook: tools.Logbook,
        halloffame: tools.HallOfFame,
        stats: tools.Statistics,
        generation: int,
    ) -> tuple[tools.Logbook, tools.HallOfFame]:
        """Evaluate the cost function and update the statistiques."""
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        if self.client is None:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
        else:
            futures_results = self.client.map(toolbox.evaluate, invalid_ind)
            fitnesses = self.client.gather(futures_results)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(individuals)

        record = stats.compile(individuals) if stats else {}
        logbook.record(gen=generation, nevals=len(invalid_ind), **record)

        return logbook, halloffame

    def _core(
        self: GeneticAlgorithm, toolbox: base.Toolbox
    ) -> tuple[Sequence[Sequence[float]], tools.Logbook, tools.HallOfFame]:
        """
        The core function as it is described in the DEAP documentation. It is adapted to allow Dask client for
        parallel computing.
        """
        halloffame, logbook, stats = self._helper_core_initialize_viewer_containt()

        population = toolbox.population(n=self.parameter_genetic_algorithm.POP_SIZE)

        logbook, halloffame = self._helper_core_evaluate(toolbox, population, logbook, halloffame, stats, generation=0)

        for gen in range(1, self.parameter_genetic_algorithm.NGEN + 1):
            offspring = toolbox.select(population, len(population))

            offspring = algorithms.varAnd(
                offspring, toolbox, self.parameter_genetic_algorithm.CXPB, self.parameter_genetic_algorithm.MUTPB
            )  # MUTATE + MATE

            logbook, halloffame = self._helper_core_evaluate(toolbox, offspring, logbook, halloffame, stats, gen)

            population[:] = offspring

        return population, logbook, halloffame

    def optimize(self: GeneticAlgorithm) -> GeneticAlgorithmViewer:
        """This is the main function. Use it to optimize your model."""
        toolbox = self.parameter_genetic_algorithm.generate_toolbox(self.parameter_optimize, self.cost_function)
        ordered_names = self.cost_function.parameters_name
        for constraint in self.constraint:
            toolbox.decorate("evaluate", constraint.generate(ordered_names))
        result = self._core(toolbox)
        return GeneticAlgorithmViewer(self.parameter_optimize, *result)
