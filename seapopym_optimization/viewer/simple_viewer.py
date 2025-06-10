"""This module contains the viwer used by the genetic_algorithm module to plot results."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from scipy.stats import entropy
from seapopym.standard.labels import ForcingLabels
from sklearn.preprocessing import QuantileTransformer

from seapopym_optimization.cost_function.observations import DayCycle, TimeSeriesObservation
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.genetic_algorithm.simple_logbook import Logbook, LogbookCategory, LogbookIndex
from seapopym_optimization.model_generator.base_model_generator import AbstractModelGenerator
from seapopym_optimization.viewer.base_viewer import AbstractViewer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from dask.distributed import Client
    from plotly.graph_objects import Figure


def compute_stats(logbook: Logbook) -> pd.DataFrame:
    """Compute the statistics of the generations."""
    stats = logbook.loc[:, LogbookCategory.WEIGHTED_FITNESS]
    stats = stats[np.isfinite(stats.loc[:, LogbookCategory.WEIGHTED_FITNESS])]
    generation_gap = (
        stats.reset_index()
        .groupby(LogbookIndex.GENERATION)[LogbookIndex.PREVIOUS_GENERATION]
        .agg(lambda x: np.sum(x) / len(x))
    )
    stats = (
        stats.groupby(LogbookIndex.GENERATION)[LogbookCategory.WEIGHTED_FITNESS]
        .aggregate(["mean", "std", "min", "max", "count"])
        .rename(columns={"count": "valid"})
    )
    stats["from_previous_generation"] = generation_gap
    return stats


@dataclass
class SimulationManager:
    """Manages a set of parameter sets and caches simulation results to avoid unnecessary recalculations."""

    param_sets: Logbook
    model_generator: AbstractModelGenerator
    functional_groups: FunctionalGroupSet
    _cache: list[xr.DataArray] = field(default_factory=list)

    def run_first(self, n: int) -> xr.DataArray:
        """Returns the results of the first n parameter sets. Missing simulations are executed and cached."""
        if not 0 <= n <= len(self.param_sets):
            msg = f"n must be between 0 and {len(self.param_sets)}"
            raise ValueError(msg)

        for i in range(len(self._cache), n):
            args = self.param_sets.iloc[[i]][LogbookCategory.PARAMETER].to_numpy()[0]
            model = self.model_generator.generate(
                functional_group_names=self.functional_groups.functional_groups_name(),
                functional_group_parameters=self.functional_groups.generate(args),
            )
            model.run()
            result = model.state[ForcingLabels.biomass]
            self._cache.append(result)

        return xr.concat(self._cache[:n], dim="individual")


@dataclass
class SimpleViewer(AbstractViewer):
    """
    Structure that contains the output of the optimization. Use the representation to plot some informations about the
    results.
    """

    observations: Sequence[TimeSeriesObservation]
    cost_function_weight: tuple[Number]

    simulation_manager: SimulationManager = field(init=False, default=None, repr=False)

    def __post_init__(self: SimpleViewer) -> None:
        """Initialize the simulation manager."""
        self.simulation_manager = SimulationManager(
            param_sets=self.hall_of_fame(drop_duplicates=True),
            model_generator=self.model_generator,
            functional_groups=self.functional_group_set,
        )

    @property
    def parameters_names(self: SimpleViewer) -> list[str]:
        """Return the names of the parameters as an ordered list."""
        return list(self.functional_group_set.unique_functional_groups_parameters_ordered().keys())

    @property
    def parameters_lower_bounds(self: SimpleViewer) -> list[float]:
        """Return the lower bounds of the parameters as an ordered list."""
        return [
            param.lower_bound
            for param in self.functional_group_set.unique_functional_groups_parameters_ordered().values()
        ]

    @property
    def parameters_upper_bound(self: SimpleViewer) -> list[float]:
        """Return the upper bounds of the parameters as an ordered list."""
        return [
            param.upper_bound
            for param in self.functional_group_set.unique_functional_groups_parameters_ordered().values()
        ]

    def stats(self: SimpleViewer) -> pd.DataFrame:
        """A review of the generations stats."""
        return compute_stats(self.logbook)

    def hall_of_fame(self: SimpleViewer, *, drop_duplicates: bool = True) -> pd.DataFrame:
        """The best individuals and their fitness."""
        logbook = self.logbook.copy()
        condition_not_inf = np.isfinite(logbook[LogbookCategory.WEIGHTED_FITNESS, LogbookCategory.WEIGHTED_FITNESS])
        logbook = logbook[condition_not_inf]
        if drop_duplicates:
            logbook = logbook.drop_duplicates(keep="first")
        return logbook.sort_values(
            (LogbookCategory.WEIGHTED_FITNESS, LogbookCategory.WEIGHTED_FITNESS), ascending=False
        )

    def fitness_evolution(self: SimpleViewer, *, points: bool = False, log_y: bool = False) -> Figure:
        """Print the evolution of the fitness by generation."""
        data = self.logbook[LogbookCategory.WEIGHTED_FITNESS].reset_index()
        data = data[np.isfinite(data[LogbookCategory.WEIGHTED_FITNESS])]
        figure = px.violin(
            data_frame=data,
            x=LogbookIndex.GENERATION,
            y=LogbookCategory.WEIGHTED_FITNESS,
            box=True,
            points=points,
            log_y=log_y,
        )

        median_values = data.groupby(LogbookIndex.GENERATION).median().reset_index()
        figure.add_scatter(
            x=median_values[LogbookIndex.GENERATION],
            y=median_values[LogbookCategory.WEIGHTED_FITNESS],
            mode="lines",
            line={"color": "rgba(0,0,0,0.5)", "width": 2, "dash": "dash"},
            name="Median",
        )

        figure.update_layout(title_text="Fitness evolution")
        return figure

    def box_plot(
        self: SimpleViewer, columns_number: int, nbest: int | None = None, *, drop_duplicates: bool = False
    ) -> go.Figure:
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

        hof = self.hall_of_fame(drop_duplicates=drop_duplicates)

        if nbest is None:
            nbest = len(hof)

        for i, (pname, lbound, ubound) in enumerate(
            zip(self.parameters_names, self.parameters_lower_bounds, self.parameters_upper_bound, strict=True)
        ):
            fig.add_trace(
                px.box(
                    data_frame=hof.iloc[:nbest][LogbookCategory.PARAMETER],
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
        self: SimpleViewer,
        *,
        nbest: int | None = None,
        uniformed: bool = False,
        parameter_groups: list[list[str]] | None = None,
        colorscale: list | str | None = None,
        unselected_opacity: float = 0.2,
    ) -> list[Figure]:
        """
        Print the `nhead` best individuals in the hall_of_fame as parallel coordinates plots for each parameter
        group.
        """
        if colorscale is None:
            colorscale = px.colors.diverging.Portland

        hof_fitness = self.hall_of_fame(drop_duplicates=True)

        if nbest is not None:
            hof_fitness = hof_fitness.iloc[:nbest]

        if uniformed:
            transformer = QuantileTransformer(output_distribution="uniform")
            hof_fitness[LogbookCategory.WEIGHTED_FITNESS] = transformer.fit_transform(
                hof_fitness[[LogbookCategory.WEIGHTED_FITNESS]]
            )

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
                    "values": hof_fitness[LogbookCategory.PARAMETER, param],
                }
                for param in group
            ]
            # NOTE(Jules): It is impossible to choose the order of Z-levels in plotly. So I use the negative fitness to
            # have the best individuals on front.
            fig = go.Figure(
                data=go.Parcoords(
                    line={
                        "color": -hof_fitness[LogbookCategory.WEIGHTED_FITNESS],
                        "colorscale": colorscale,
                        "showscale": True,
                        "colorbar": {
                            "title": "Cost function score",
                            "tickvals": [
                                -hof_fitness[LogbookCategory.WEIGHTED_FITNESS].min(),
                                -hof_fitness[LogbookCategory.WEIGHTED_FITNESS].max(),
                            ],
                            "tickmode": "array",
                            "ticktext": [
                                hof_fitness[LogbookCategory.WEIGHTED_FITNESS].min(),
                                hof_fitness[LogbookCategory.WEIGHTED_FITNESS].max(),
                            ],
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
                coloraxis_colorbar={"title": "Fitness (uniforme distribution)" if uniformed else "Cost function score"},
                title_text=(
                    "Parameters optimization : minimization of the cost function for group "
                    f"{parameter_groups.index(group) + 1}"
                ),
            )
            figures.append(fig)

        return figures

    def parameters_standardized_deviation(self: SimpleViewer) -> go.Figure:
        """Print the standardized deviation of the parameters by generation."""
        param_range = pd.Series(
            np.subtract(self.parameters_upper_bound, self.parameters_lower_bounds), index=self.parameters_names
        )
        param_std = (
            self.logbook[LogbookCategory.PARAMETER]
            .droplevel(level=[LogbookIndex.PREVIOUS_GENERATION.get_index(), LogbookIndex.INDIVIDUAL.get_index()])
            .reset_index()
            .groupby(LogbookIndex.GENERATION)
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

    def shannon_entropy(self: SimpleViewer, *, bins: int = 10) -> go.Figure:
        """Proche de 0 = distribution similaires."""

        def compute_shannon_entropy(p: np.ndarray) -> float:
            """Close to 0 = similar distribution."""
            hist_p, _ = np.histogram(p, bins=bins, density=True)
            hist_p += 1e-10
            return entropy(hist_p / np.sum(hist_p))

        data = self.logbook[LogbookCategory.PARAMETER].reset_index()

        entropies = {}
        for generation in data[LogbookIndex.GENERATION].unique():
            data_gen = data[data[LogbookIndex.GENERATION] == generation]
            gen_entropy = {k: compute_shannon_entropy(v) for k, v in data_gen.items() if k in self.parameters_names}
            entropies[generation] = gen_entropy

        entropies = pd.DataFrame(entropies).T

        entropies = (
            entropies.unstack()
            .reset_index()
            .rename(columns={"level_1": "Generation", "level_0": "Variable", 0: "Entropy"})
        )

        return px.area(
            entropies,
            x="Generation",
            y="Entropy",
            color="Variable",
            line_group="Variable",
            title="Shannon entropy of parameter distributions",
            labels={"index": "Generation", "value": "Shannon entropy"},
            markers=True,
        ).update_layout(xaxis_showgrid=False, yaxis_showgrid=False, plot_bgcolor="rgba(0, 0, 0, 0)")

    def time_series(self: SimpleViewer, nbest: int, title: Iterable[str] | None = None) -> list[go.Figure]:
        """Plot the time series of the best simulations for each observation."""

        def _plot_observation(observation: xr.DataArray, day_cycle: str, layer: int) -> go.Scatter:
            y = observation.sel(layer=layer)
            x = y.cf["T"]
            return go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=f"Observed {day_cycle} layer {layer}",
                line={"dash": "dash", "width": 1, "color": "black"},
                marker={"size": 4, "symbol": "x", "color": "black"},
            )

        def _compute_fgroup_in_layer(day_cycle: DayCycle, layer: int) -> list[int]:
            return [
                fg_index
                for fg_index, fg in enumerate(self.functional_group_set.functional_groups)
                if (fg.night_layer == layer and day_cycle == DayCycle.DAY)
                or (fg.day_layer == layer and day_cycle == DayCycle.Night)
            ]

        def _plot_best_prediction(
            prediction: xr.DataArray, fgroup: Iterable[int], day_cycle: DayCycle, layer: int
        ) -> go.Scatter:
            y = prediction.sel(functional_group=fgroup, individual=0).sum("functional_group")
            x = y.cf["T"]
            return go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=f"Predicted {day_cycle} layer {layer}",
                line={
                    "dash": "solid",
                    "width": 2,
                    "color": "royalblue" if day_cycle == "night" else "firebrick",
                },
            )

        def _plot_range_best_predictions(
            prediction: xr.DataArray, fgroup: Iterable[int], nbest: int, day_cycle: DayCycle, layer: int
        ) -> None:
            y = prediction.sel(functional_group=fgroup).sum("functional_group")  # total biomass in layer
            x = prediction.time.to_series()
            x_rev = pd.concat([x, x[::-1]])
            y_upper = y.max("individual")
            y_lower = y.min("individual")[::-1]
            y_rev = np.concatenate([y_upper, y_lower])

            return go.Scatter(
                x=x_rev,
                y=y_rev,
                fill="toself",
                line_color="rgba(0,0,0,0)",
                fillcolor="rgba(54,92,216,0.3)" if day_cycle == "night" else "rgba(174,30,36,0.3)",
                name=f"{day_cycle} layer {layer} : {nbest} best individuals",
            )

        best_simulations = self.simulation_manager.run_first(nbest)
        best_simulations = best_simulations.pint.quantify().pint.to("milligram / meter ** 2").pint.dequantify()

        nb_columns = 2  # day, night
        layer_pos = np.ravel([(fg.day_layer, fg.night_layer) for fg in self.functional_group_set.functional_groups])
        upper_layer_pos = layer_pos.min()
        lower_layer_pos = layer_pos.max()
        nb_rows = int(lower_layer_pos - upper_layer_pos + 1)

        all_figures = []
        for fig_nb, observation in enumerate(self.observations):
            figure = make_subplots(
                rows=nb_rows,
                cols=nb_columns,
                x_title="Time",
                y_title="Biomass (mg/m2)",
                row_titles=[f"Layer {layer}" for layer in np.sort(np.unique(layer_pos))],
                subplot_titles=["Day", "Night"],
                horizontal_spacing=0.1,
                vertical_spacing=0.1,
            )
            obs_data: xr.DataArray = (
                observation.observation.pint.quantify().pint.to("milligram / meter ** 2").pint.dequantify()
            )
            best_simulations_sel = best_simulations.cf.sel(X=obs_data.cf["X"], Y=obs_data.cf["Y"]).cf.mean(["X", "Y"])
            obs_data = obs_data.cf.mean(["X", "Y"])
            column = 1 if observation.observation_type == DayCycle.DAY else 2

            for layer in np.unique(layer_pos):
                row = int(layer - upper_layer_pos) + 1  # 1-indexed
                fgroup = _compute_fgroup_in_layer(observation.observation_type, layer)
                if len(fgroup) > 0:
                    figure.add_trace(
                        _plot_best_prediction(best_simulations_sel, fgroup, observation.observation_type, layer),
                        row=row,
                        col=column,
                    )
                    figure.add_trace(
                        _plot_range_best_predictions(
                            best_simulations_sel, fgroup, nbest, observation.observation_type, layer
                        ),
                        row=row,
                        col=column,
                    )

                if layer in obs_data.layer and obs_data.sel(layer=layer).sum() > 0:
                    figure.add_trace(
                        _plot_observation(obs_data, observation.observation_type, layer), row=row, col=column
                    )
            figure.update_layout(title=f"{title[fig_nb]}" if title is not None else observation.name)
            all_figures.append(figure)

        return all_figures

    # ---------------------------------------------------------------------------------------------------------------- #
    # TODO(Jules): Finish adaptation of following functions
    # ---------------------------------------------------------------------------------------------------------------- #

    def parameters_scatter_matrix(
        self: SimpleViewer,
        nbest: int | None = None,
        size: int = 1000,
        **kwargs: dict,
    ) -> go.Figure:
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
            height=size,
            width=size,
            color=LogbookCategory.WEIGHTED_FITNESS,
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

    def parameters_correlation_matrix(self: SimpleViewer, nbest: int | None = None) -> go.Figure:
        """Print the correlation matrix of the parameters for the N best individuals."""
        indiv_param = self.hall_of_fame.iloc[:nbest, :-1].to_numpy()
        param_names = self.hall_of_fame.columns[:-1]

        corr_matrix = np.corrcoef(indiv_param.T)
        np.fill_diagonal(corr_matrix, np.nan)

        fig = px.imshow(
            corr_matrix,
            text_auto=False,
            aspect="auto",
            color_continuous_scale=[[0, "blue"], [0.5, "white"], [1, "red"]],
            zmin=-1,
            zmax=1,
            x=param_names,
            y=param_names,
        )
        fig.update_layout(
            title=f"Correlation Matrix of {nbest} individuals",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis={"showgrid": False, "tickangle": -35},
            yaxis={"showgrid": False},
        )
        return fig

    # TODO(Jules) : Be able to zoom correlation axis. Like corr_range=[0.7, 1]
    def taylor_diagram(self: SimpleViewer, nbest: int, client=None, range_theta: Iterable[int] = [0, 90]) -> go.Figure:
        best_simulations = self.best_individuals_simulations(nbest, client=client)

        day_layer = np.array([fg.day_layer for fg in self.parameters.functional_groups])
        night_layer = np.array([fg.night_layer for fg in self.parameters.functional_groups])

        data = {
            "name": [],
            "color": [],
            "correlation_coefficient": [],
            "normalized_root_mean_square_error": [],
            "normalized_standard_deviation": [],
            # "bias": [],
        }

        day_color = "firebrick"
        night_color = "royalblue"

        for observation in self.observations:
            for individual in best_simulations[LogbookIndex.INDIVIDUAL].data:
                # TODO(Jules): Add day / night as differents individuals
                prediction = best_simulations.sel(individual=individual)

                corr_day, corr_night = observation.correlation_coefficient(prediction, day_layer, night_layer)
                mse_day, mse_night = observation.mean_square_error(
                    prediction, day_layer, night_layer, root=True, normalized=True
                )
                std_day, std_night = observation.normalized_standard_deviation(prediction, day_layer, night_layer)
                # bias = observation.bias(prediction, day_layer, night_layer, standardize=True)
                # DAY
                data["name"].append(f"{observation.name} x Individual {individual} x Day")
                data["color"].append(day_color)
                data["correlation_coefficient"].append(np.float64(corr_day))
                data["normalized_root_mean_square_error"].append(np.float64(mse_day))
                data["normalized_standard_deviation"].append(np.float64(std_day))
                # NIGHT
                data["name"].append(f"{observation.name} x Individual {individual} x Night")
                data["color"].append(night_color)
                data["correlation_coefficient"].append(np.float64(corr_night))
                data["normalized_root_mean_square_error"].append(np.float64(mse_night))
                data["normalized_standard_deviation"].append(np.float64(std_night))

        data["angle"] = np.asarray(data["correlation_coefficient"]) * 90
        data = pd.DataFrame(data).dropna(axis=0)

        fig = px.scatter_polar(
            data,
            r="normalized_standard_deviation",
            theta="angle",
            color="color",
            symbol="name",
            # color_discrete_sequence=px.colors.sequential.Plasma_r,
            start_angle=90,
            range_theta=range_theta,
            direction="clockwise",  # Change direction to clockwise
            range_r=[0, 2],
            custom_data=[
                "name",
                "correlation_coefficient",
                "normalized_standard_deviation",
                # "bias",
                "normalized_root_mean_square_error",
            ],
            title="Taylor diagram",
        )

        fig.update_traces(
            marker={
                "size": 10,
                # add contour line around markers
                "line": {"color": "black", "width": 1},
                # change opacity
                "opacity": 0.8,
            },
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>"
                "Correlation: %{customdata[1]:.2f}<br>"
                "Normalized STD: %{customdata[2]:.2f}<br>"
                # "Bias: %{customdata[3]:.2f}<br>"
                "Normalized Bias: %{customdata[4]:.2f}<br>",
            ),
        )

        angles = np.linspace(-90, 90, 90)
        r_cercle = np.full_like(angles, 1)
        fig.add_trace(
            go.Scatterpolar(
                r=r_cercle,
                theta=angles,
                mode="lines",
                line={"color": "red", "width": 2, "dash": "dash"},
                hoverinfo="skip",
                showlegend=False,
            ),
        )

        fig.update_layout(
            coloraxis_colorbar={
                "title": "Bias",
                "title_side": "top",
                "orientation": "h",
                "len": 0.7,
                "yanchor": "top",  # Le haut de la colorbar est en position -0.1
                "y": -0.1,
                "xanchor": "center",  # le centre de la colorbar est en position 0.5
                "x": 0.5,
            },
            legend={
                "xanchor": "right",
                "yanchor": "top",
                "x": 1,
                "y": 1,
                "title": "Station x Day/Night",
            },
            height=800,
            margin={"l": 100, "r": 100, "t": 100, "b": 100},
            polar={
                "angularaxis": {
                    "dtick": 9,
                    "tickmode": "array",
                    "tickvals": np.arange(-90, 91, 9),
                    "ticktext": [f"{i:.1f}" for i in np.arange(-1, 0, 0.1)]
                    + [f"{i:.1f}" for i in np.arange(0, 1.01, 0.1)],
                }
            },
        )

        return fig
