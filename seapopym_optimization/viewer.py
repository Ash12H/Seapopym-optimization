"""This module contains the viwer used by the genetic_algorithm module to plot results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

from seapopym_optimization import wrapper

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objects import Figure
    from seapopym.configuration.no_transport.parameter import ForcingParameters

    from seapopym_optimization.cost_function import Observation
    from seapopym_optimization.functional_groups import AllGroups


def _compute_stats(logbook: pd.DataFrame) -> pd.DataFrame:
    """Compute the statistics of the generations."""
    stats = logbook[np.isfinite(logbook["fitness_final"])]
    generation_gap = stats.reset_index().groupby("generation")["previous_generation"].agg(lambda x: np.sum(x) / len(x))
    stats = (
        stats.groupby("generation")["fitness_final"]
        .aggregate(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "valid"})
    )
    stats["from_previous_generation"] = generation_gap
    return stats


@dataclass
class GeneticAlgorithmViewer:
    """
    Structure that contains the output of the optimization. Use the representation to plot some informations about the
    results.
    """

    logbook: pd.DataFrame
    parameters: AllGroups
    forcing_parameters: ForcingParameters
    observations: Sequence[Observation]
    _minimize: bool = field(init=False)

    def __post_init__(self: GeneticAlgorithmViewer) -> None:
        """Check the logbook and set the minimize attribute."""
        self._minimize = self.logbook["fitness_final"].max() < 0

        self.logbook = self.logbook.drop(columns=["fitness"]).rename(columns={"fitness_final": "fitness"})
        self.logbook["fitness"] = self.logbook["fitness"].abs()

    @property
    def parameters_names(self: GeneticAlgorithmViewer) -> list[str]:
        return list(self.parameters.unique_functional_groups_parameters_ordered.keys())

    @property
    def parameters_lower_bounds(self: GeneticAlgorithmViewer):
        return [param.lower_bound for param in self.parameters.unique_functional_groups_parameters_ordered.values()]

    @property
    def parameters_upper_bound(self: GeneticAlgorithmViewer):
        return [param.upper_bound for param in self.parameters.unique_functional_groups_parameters_ordered.values()]

    @property
    def stats(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """A review of the generations stats."""
        return _compute_stats(self.logbook)

    @property
    def hall_of_fame(self: GeneticAlgorithmViewer) -> pd.DataFrame:
        """The best individuals and their fitness."""
        logbook = self.logbook
        condition_not_inf = np.isfinite(logbook["fitness"])
        condition_not_already_calculated = ~logbook.index.get_level_values("previous_generation")
        condition = condition_not_inf & condition_not_already_calculated
        previous_generation_level = 1
        return logbook[condition].sort_values("fitness", ascending=self._minimize).droplevel(previous_generation_level)

    def best_individuals_simulations(self: GeneticAlgorithmViewer, nbest: int | None = None) -> go.Figure:
        biomass_accumulated = []
        for cpt, (_, individual_parameters) in enumerate(self.hall_of_fame[:nbest].iterrows()):
            individual = self.parameters.generate_matrix(
                [individual_parameters[name] for name in self.parameters_names]
            )
            model = wrapper.model_generator_no_transport(
                forcing_parameters=self.forcing_parameters,
                fg_parameters=wrapper.FunctionalGroupGeneratorNoTransport(
                    parameters=individual, groups_name=self.parameters.functional_groups_name
                ),
            )
            model.run()

            biomass_accumulated.append(model.export_biomass().expand_dims({"individual": [cpt]}))
        return xr.concat(biomass_accumulated, dim="individual")

    def fitness_evolution(self: GeneticAlgorithmViewer, *, log_y: bool = True) -> Figure:
        """Print the evolution of the fitness by generation."""
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

    def box_plot(self: GeneticAlgorithmViewer, columns_number: int, nbest: int | None = None) -> go.Figure:
        """Print the `nbest` best individuals in the hall_of_fame as box plots."""
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
        *,
        nbest: int | None = None,
        parameter_groups: list[list[str]] | None = None,
        colorscale: list | str | None = None,
        unselected_opacity: float = 0.2,
    ) -> list[Figure]:
        """Print the `nhead` best individuals in the hall_of_fame as parallel coordinates plots for each parameter group."""
        if colorscale is None:
            colorscale = [
                [0, "rgba(0, 0, 255, 0.8)"],
                [0.3, "rgba(255,255,0,0.5)"],
                [1.0, "rgba(255,255,0,0.0)"],
            ]

        hof_fitness = self.hall_of_fame

        if nbest is not None:
            hof_fitness = hof_fitness.iloc[:nbest]

        if parameter_groups is None:
            parameter_groups = [self.parameters_names]

        figures = []

        for group in parameter_groups:
            dimensions = [
                {
                    "range": [
                        self.parameters_lower_bounds[self.parameters_names.index(param)],
                        self.parameters_upper_bound[self.parameters_names.index(param)],
                    ],
                    "label": param,
                    "values": hof_fitness[param],
                }
                for param in group
            ]
            # NOTE(Jules): It is impossible to choose the order of Z-levels in plotly. So I use the negative fitness to
            # have the best individuals on front.
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
                title_text=f"Parameters optimization : minimization of the cost function for group {parameter_groups.index(group) + 1}",
            )
            figures.append(fig)

        return figures

    def parameters_standardized_deviation(self: GeneticAlgorithmViewer) -> go.Figure:
        """Print the standardized deviation of the parameters by generation."""
        param_range = {
            name: ub - lb
            for name, ub, lb in zip(self.parameters_names, self.parameters_upper_bound, self.parameters_lower_bounds)
        }
        param_std = (
            self.logbook.reset_index()
            .drop(columns=["previous_generation", "individual", "fitness"])
            .groupby("generation")
            .std()
        )
        param_standardized_std = param_std / param_range

        fig = make_subplots(
            rows=1,
            cols=len(param_standardized_std.index),
            shared_yaxes=True,
            subplot_titles=[f"Gen={i}" for i in param_standardized_std.index],
        )

        for generation in param_standardized_std.index:
            fig.add_trace(
                go.Bar(
                    y=param_standardized_std.columns,
                    x=param_standardized_std.loc[generation],
                    orientation="h",
                ),
                row=1,
                col=generation + 1,
            )

        fig.update_layout(
            title="Standardized std of parameters by generation",
            showlegend=False,
        )
        return fig

    def parameters_scatter_matrix(self: GeneticAlgorithmViewer, nbest: int | None = None, **kwargs: dict) -> go.Figure:
        """
        Print the scatter matrix of the parameters.
        Usefull to explore wich combination of parameters are used and if the distribution is correct.
        """
        data = self.hall_of_fame
        if nbest is not None:
            data = data[:nbest]

        fig = px.scatter_matrix(
            data,
            dimensions=data.columns[:-1],
            height=1500,
            width=1500,
            color="fitness",
            color_continuous_scale=[
                (0, "rgba(0,0,255,1)"),
                (0.3, "rgba(255,0,0,0.8)"),
                (1, "rgba(255,255,255,0.0)"),
            ],
            **kwargs,
        )

        fig.update_traces(marker={"size": 3}, unselected={"marker": {"opacity": 0.01}})

        param_bounds = {
            name: (lb, ub)
            for name, lb, ub in zip(self.parameters_names, self.parameters_lower_bounds, self.parameters_upper_bound)
        }

        for i, param_name in enumerate(data.columns[:-1]):
            lower_bound = param_bounds[param_name][0]
            upper_bound = param_bounds[param_name][1]
            fig.update_xaxes(range=[lower_bound, upper_bound], row=i + 1, col=i + 1)

        return fig
