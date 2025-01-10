"""This module contains the main genetic algorithm functions that can be used to optimize the model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from deap import algorithms, base, tools
from IPython.display import clear_output, display
from plotly.subplots import make_subplots
from tqdm import tqdm

if TYPE_CHECKING:
    from dask.distributed import Client
    from plotly.graph_objects import Figure

    from seapopym_optimization.constraint import GenericConstraint
    from seapopym_optimization.cost_function import GenericCostFunction, Parameter


def _compute_stats(logbook: pd.DataFrame) -> pd.DataFrame:
    """Compute the statistics of the generations."""
    stats = logbook[np.isfinite(logbook["fitness"])]
    generation_gap = stats.reset_index().groupby("generation")["previous_generation"].agg(lambda x: np.sum(x) / len(x))
    stats = (
        stats.groupby("generation")["fitness"]
        .aggregate(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "valid"})
    )
    stats["from_previous_generation"] = generation_gap
    return stats


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


@dataclass
class GeneticAlgorithmViewer:
    """
    Structure that contains the output of the optimization. Use the representation to plot some informations about the
    results.
    """

    _parameters: Sequence[Parameter]
    _logbook: pd.DataFrame

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
    def stats(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """A review of the generations stats."""
        return _compute_stats(self.logbook)

    @property
    def logbook(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """A review of the generations stats."""
        return self._logbook.copy()

    @property
    def hall_of_fame(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """The best individuals and their fitness."""
        logbook = self.logbook
        condition_not_inf = np.isfinite(logbook["fitness"])
        condition_not_already_calculated = ~logbook.index.get_level_values("previous_generation")
        condition = condition_not_inf & condition_not_already_calculated
        previous_generation_level = 1
        return logbook[condition].sort_values("fitness", ascending=True).droplevel(previous_generation_level)

    def fitness_evolution(self: GeneticAlgorithmViewer, log_y: bool = True) -> Figure:
        data = self.logbook[np.isfinite(self.logbook["fitness"])]["fitness"].reset_index()
        figure = px.box(data_frame=data, x="generation", y="fitness", points=False, log_y=log_y)

        median_values = data.groupby("generation").median().reset_index()
        figure.add_scatter(
            x=median_values["generation"],
            y=median_values["fitness"],
            mode="lines",
            line={"color": "rgba(0,0,0,0.5)", "width": 2, "dash": "dash"},
            name="Median",
        )

        min_values = data.groupby("generation").min().reset_index()
        figure.add_scatter(
            x=min_values["generation"],
            y=min_values["fitness"],
            mode="lines",
            line={"color": "rgba(0,0,0,0.5)", "width": 2, "dash": "dash"},
            name="Minimum",
        )

        figure.update_layout(title_text="Fitness evolution")
        return figure

    def box_plot(self: GeneticAlgorithmViewer, columns_number: int, nbest: int | None = None):
        nb_fig = len(self.parameters_names)
        nb_row = nb_fig // columns_number + (1 if nb_fig % columns_number > 0 else 0)

        fig = make_subplots(
            rows=nb_row,
            cols=columns_number,
            subplot_titles=self.parameters_names,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
        )

        if nbest is None:
            nbest = len(self.hall_of_fame)

        for i, (pname, lbound, ubound) in enumerate(
            zip(self.parameters_names, self.parameters_lower_bounds, self.parameters_upper_bound)
        ):
            fig.add_trace(
                px.box(
                    data_frame=self.hall_of_fame[:nbest],
                    y=pname,
                    range_y=[lbound, ubound],  # Not working with "add_trace" function.
                    title=pname,
                ).data[0],
                row=(i // columns_number) + 1,
                col=(i % columns_number) + 1,
            )

        fig.update_layout(
            title_text="Parameters distribution",
            height=nb_row * 300,
            width=columns_number * 300,
        )

        return fig

    def parallel_coordinates(
        self: GeneticAlgorithmViewer,
        nbest: int | None = None,
        colorscale: list | str | None = None,
        unselected_opacity: float = 0.2,
    ) -> Figure:
        """Print the `nhead` best individuals in the hall_of_fame as a parallel coordinates plot."""
        if colorscale is None:
            colorscale = [
                [0, "rgba(0, 0, 255, 0.8)"],
                [0.3, "rgba(255,255,0,0.5)"],
                [1.0, "rgba(255,255,0,0.0)"],
            ]

        hof_fitness = self.hall_of_fame

        if nbest is not None:
            hof_fitness = hof_fitness.iloc[:nbest]

        dimensions = [
            {
                "range": [self.parameters_lower_bounds[i], self.parameters_upper_bound[i]],
                "label": self.parameters_names[i],
                "values": hof_fitness[self.parameters_names[i]],
            }
            for i in range(len(self.parameters_names))
        ]

        # NOTE(Jules): We reversed the values because Plotly set the individuals with the maximum value at the front.
        # As we minimize the function, we want the individuals with the minimum value at the front for a better
        # visualization.
        fig = go.Figure(
            data=go.Parcoords(
                line={
                    "color": -hof_fitness["fitness"],
                    "colorscale": colorscale,
                    "showscale": True,
                    "colorbar": {
                        "title": "Cost function score",
                        "tickvals": [-hof_fitness["fitness"].min(), -hof_fitness["fitness"].max()],
                        "tickmode": "array",
                        "ticktext": [hof_fitness["fitness"].min(), hof_fitness["fitness"].max()],
                    },
                    "reversescale": True,
                },
                dimensions=dimensions,
                unselected={
                    "line": {"opacity": unselected_opacity},
                },
            )
        )

        fig.update_layout(
            coloraxis_colorbar={"title": "Fitness"},
            title_text="Parameters optimization : minimization of the cost function",
        )
        return fig


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

    def __post_init__(self: GeneticAlgorithm) -> None:
        """Check parameters."""
        # TODO(Jules): Vérifier que les paramètres ont des noms uniques.

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
        """Evaluate the cost function and update the statistiques."""
        known = [ind.fitness.valid for ind in individuals]
        invalid_ind = [ind for ind in individuals if not ind.fitness.valid]
        if self.client is None:
            fitnesses = list(map(toolbox.evaluate, invalid_ind))
        else:
            futures_results = self.client.map(toolbox.evaluate, invalid_ind)
            fitnesses = self.client.gather(futures_results)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        df_logbook = pd.DataFrame(
            individuals, columns=self.cost_function.functional_groups.unique_functional_groups_parameters_ordered.keys()
        )
        df_logbook["fitness"] = [ind.fitness.values[0] for ind in individuals]
        df_logbook["previous_generation"] = known
        df_logbook["generation"] = generation
        df_logbook.index.name = "individual"
        df_logbook = df_logbook.reset_index()
        return df_logbook.set_index(["generation", "previous_generation", "individual"]).sort_index()

    def _core(
        self: GeneticAlgorithm, toolbox: base.Toolbox
    ) -> tuple[Sequence[Sequence[float]], tools.Logbook, tools.HallOfFame]:
        """
        The core function as it is described in the DEAP documentation. It is adapted to allow Dask client for
        parallel computing. The order used is SCM: Select, Cross, Mutate.
        """
        for gen in tqdm(desc="Generations", iterable=range(self.parameter_genetic_algorithm.NGEN + 1)):
            if gen == 0:
                population = toolbox.population(n=self.parameter_genetic_algorithm.POP_SIZE)

                df_logbook = self._helper_core_evaluate(toolbox, population, gen)
            else:
                offspring = toolbox.select(population, len(population))
                offspring = algorithms.varAnd(
                    offspring, toolbox, self.parameter_genetic_algorithm.CXPB, self.parameter_genetic_algorithm.MUTPB
                )  # MUTATE + MATE

                df_logbook = pd.concat([df_logbook, self._helper_core_evaluate(toolbox, offspring, gen)])

                # TODO(Jules): L'intégralité de la population est remplacée par les nouveaux individus. Je pourrai ajouter
                # paramètre GGAP pour conserver les meilleurs individus de la génération précédente.
                # Cf. Maria Angelova and Tania Pencheva 2011 - Tuning Genetic Algorithm Parameters to Improve Convergence
                # Time

                population[:] = offspring
            clear_output(wait=True)
            display(_compute_stats(df_logbook))

        return df_logbook

    def optimize(self: GeneticAlgorithm) -> GeneticAlgorithmViewer:
        """This is the main function. Use it to optimize your model."""
        ordered_parameters = self.cost_function.functional_groups.unique_functional_groups_parameters_ordered
        toolbox = self.parameter_genetic_algorithm.generate_toolbox(ordered_parameters.values(), self.cost_function)
        for constraint in self.constraint:
            toolbox.decorate("evaluate", constraint.generate(list(ordered_parameters.keys())))
        result = self._core(toolbox)
        return GeneticAlgorithmViewer(
            self.cost_function.functional_groups.unique_functional_groups_parameters_ordered.values(), result
        )
