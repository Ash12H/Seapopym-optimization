"""This module contains the viwer used by the genetic_algorithm module to plot results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    import pandas as pd
    from plotly.graph_objects import Figure

    from seapopym_optimization.functional_groups import Parameter


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
