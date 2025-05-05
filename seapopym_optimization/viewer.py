"""This module contains the viwer used by the genetic_algorithm module to plot results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots
from scipy.stats import entropy

from seapopym_optimization import wrapper

if TYPE_CHECKING:
    from dask.distributed import Client
    from plotly.graph_objects import Figure
    from seapopym.configuration.no_transport.parameter import ForcingParameters

    from seapopym_optimization.cost_function import Observation
    from seapopym_optimization.functional_groups import AllGroups
from sklearn.preprocessing import QuantileTransformer


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
    _minimize: bool = field(init=False, default=None)
    _nbest_simulations: xr.Dataset = field(init=False, default=None)

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
        # In case of constraint violation:
        condition_not_inf = np.isfinite(logbook["fitness"])
        # Avoid to take the same individual:
        condition_not_already_calculated = ~logbook.index.get_level_values("previous_generation")
        condition = condition_not_inf & condition_not_already_calculated
        previous_generation_level = 1
        return logbook[condition].sort_values("fitness", ascending=self._minimize).droplevel(previous_generation_level)

    @property
    def original_simulation(self: GeneticAlgorithmViewer) -> xr.Dataset:
        original_config = [[0, 0, 0.1668, 10.38, -0.11, 150, -0.15]]
        original_model = wrapper.model_generator_no_transport(
            forcing_parameters=self.forcing_parameters,
            fg_parameters=wrapper.FunctionalGroupGeneratorNoTransport(
                parameters=original_config, groups_name=["Total"]
            ),
        )

        original_model.run()
        return original_model.export_biomass()
    
    @property
    def original_simulation_acidity(self: GeneticAlgorithmViewer) -> xr.Dataset:
        original_config = [[0, 0, 0.01668, 41.61, -0.085, 0.04, 0.004, 2.4, -0.58]]
        original_model = wrapper.model_generator_acidity(
            forcing_parameters=self.forcing_parameters,
            fg_parameters=wrapper.FunctionalGroupGeneratorAcidity(
                parameters=original_config, groups_name=["Total"]
            ),
        )

        original_model.run()
        return original_model.export_biomass()

    @property
    def best_simulation(self: GeneticAlgorithmViewer) -> xr.Dataset:
        if self._nbest_simulations is not None:
            return self._nbest_simulations.sel(individual=0)

        return self.best_individuals_simulations(nbest=1)

    def best_individuals_simulations(
        self: GeneticAlgorithmViewer, nbest: int, client: Client | None = None
    ) -> xr.Dataset:
        min_nbest = 0
        if self._nbest_simulations is not None:
            if nbest <= self._nbest_simulations.sizes["individual"]:
                return self._nbest_simulations.sel(individual=slice(None, nbest - 1))

            min_nbest = self._nbest_simulations.sizes["individual"]

        individuals_parameterization = []
        for cpt, (_, individual_parameters) in enumerate(self.hall_of_fame[min_nbest:nbest].iterrows()):
            individuals_parameterization.append(
                (
                    min_nbest + cpt,
                    self.parameters.generate_matrix([individual_parameters[name] for name in self.parameters_names]),
                )
            )

        # def run_simulation(individual: tuple[int, np.ndarray]):
        #     """Take an individual as (number, parameters) and run the simulation."""
        #     model = wrapper.model_generator_no_transport(
        #         forcing_parameters=self.forcing_parameters,
        #         fg_parameters=wrapper.FunctionalGroupGeneratorNoTransport(
        #             parameters=individual[1], groups_name=self.parameters.functional_groups_name
        #         ),
        #     )
        #     model.run()
        #     return model.export_biomass().expand_dims({"individual": [individual[0]]})

        def run_simulation(individual: tuple[int, np.ndarray]):
            """Take an individual as (number, parameters) and run the simulation, either a NoTransport Simulation or an Acidity one"""
            if individual[1].shape[1]==7: # NoTransport model, 7 params to optimise
                model = wrapper.model_generator_no_transport(
                    forcing_parameters=self.forcing_parameters,
                    fg_parameters=wrapper.FunctionalGroupGeneratorNoTransport(
                        parameters=individual[1], groups_name=self.parameters.functional_groups_name
                    ),
                )
                model.run()
                return model.export_biomass().expand_dims({"individual": [individual[0]]}) 
            elif individual[1].shape[1]==9: # Acidity model, 9 params to optimise
                model = wrapper.model_generator_acidity(
                    forcing_parameters=self.forcing_parameters,
                    fg_parameters=wrapper.FunctionalGroupGeneratorAcidity(
                        parameters=individual[1], groups_name=self.parameters.functional_groups_name
                    ),
                )
                model.run()
                return model.export_biomass().expand_dims({"individual": [individual[0]]})  
            else:
                raise ValueError("Unexpected number of parameters. NoTransport model expect 7 parameters, Acidity model expect 9 parameters.")     

        if client is None:
            biomass_accumulated = [run_simulation(individual) for individual in individuals_parameterization]
        else:
            biomass_accumulated = client.map(run_simulation, individuals_parameterization)
            biomass_accumulated = client.gather(biomass_accumulated)

        if self._nbest_simulations is not None:
            self._nbest_simulations = xr.concat([self._nbest_simulations, *biomass_accumulated], dim="individual")
        else:
            self._nbest_simulations = xr.concat(biomass_accumulated, dim="individual")

        return self._nbest_simulations

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

        hof_fitness = self.hall_of_fame

        if nbest is not None:
            hof_fitness = hof_fitness.iloc[:nbest]

        if uniformed:
            transformer = QuantileTransformer(output_distribution="uniform")
            hof_fitness["fitness"] = transformer.fit_transform(hof_fitness[["fitness"]])

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
                coloraxis_colorbar={"title": "Fitness (uniforme distribution)" if uniformed else "Cost function score"},
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

    def shannon_entropy(self: GeneticAlgorithmViewer, *, bins: int = 10) -> go.Figure:
        """Proche de 0 = distribution similaires."""

        def compute_shannon_entropy(p: np.ndarray) -> float:
            """Close to 0 = similar distribution."""
            hist_p, _ = np.histogram(p, bins=bins, density=True)
            hist_p += 1e-10
            return entropy(hist_p / np.sum(hist_p))

        data = self.logbook.reset_index()

        entropies = {}
        for generation in data["generation"].unique():
            data_gen = data[data["generation"] == generation]
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

    def parameters_scatter_matrix(
        self: GeneticAlgorithmViewer,
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

    def parameters_correlation_matrix(self: GeneticAlgorithmViewer, nbest: int | None = None) -> go.Figure:
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

    def time_series(
        self: GeneticAlgorithmViewer, nbest: int, title: Iterable[str] | None = None, client=None
    ) -> list[go.Figure]:
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

        def _compute_fgroup_in_layer(day_cycle: str, layer: int) -> list[int]:
            return [
                fg_index
                for fg_index, fg in enumerate(self.parameters.functional_groups)
                if (fg.night_layer == layer and day_cycle == "night") or (fg.day_layer == layer and day_cycle == "day")
            ]

        def _plot_best_prediction(
            prediction: xr.DataArray, fgroup: Iterable[int], day_cycle: str, layer: int
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
            prediction: xr.DataArray, fgroup: Iterable[int], nbest: int, day_cycle: str, layer: int
        ) -> None:
            y = prediction.sel(functional_group=fgroup).sum("functional_group")
            x = best_simulations.time.to_series()
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

        best_simulations = self.best_individuals_simulations(nbest, client=client)
        best_simulations = best_simulations.pint.quantify().pint.to("milligram / meter ** 2")

        # ------------------------------------------------------------------------------------------------------------ #

        # ------------------------------------------------------------------------------------------------------------ #

        # ------------------------------------------------------------------------------------------------------------ #

        nb_columns = 2  # day, night
        layer_pos = np.ravel([(fg.night_layer, fg.day_layer) for fg in self.parameters.functional_groups])
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
            obs_data: xr.Dataset = observation.observation.pint.quantify().pint.to("milligram / meter ** 2")
            best_simulations_sel = best_simulations.cf.sel(X=obs_data.cf["X"], Y=obs_data.cf["Y"]).cf.mean(["X", "Y"])
            obs_data = obs_data.cf.mean(["X", "Y"])

            for column, day_cycle in enumerate(["day", "night"]):
                col = column + 1  # 1-indexed
                if day_cycle in obs_data:
                    obs_data_selected = obs_data[day_cycle]
                    for layer in np.unique(layer_pos):
                        row = int(layer - upper_layer_pos) + 1  # 1-indexed
                        fgroup = _compute_fgroup_in_layer(day_cycle, layer)
                        if len(fgroup) > 0:
                            figure.add_trace(
                                _plot_best_prediction(best_simulations_sel, fgroup, day_cycle, layer), row=row, col=col
                            )
                            figure.add_trace(
                                _plot_range_best_predictions(best_simulations_sel, fgroup, nbest, day_cycle, layer),
                                row=row,
                                col=col,
                            )
                        if layer in obs_data_selected.layer:
                            figure.add_trace(_plot_observation(obs_data_selected, day_cycle, layer), row=row, col=col)

                else:
                    pass
            figure.update_layout(title=f"{title[fig_nb]}")
            all_figures.append(figure)

        return all_figures

    # TODO(Jules) : Be able to zoom correlation axis. Like corr_range=[0.7, 1]
    def taylor_diagram(
        self: GeneticAlgorithmViewer, nbest: int = 1, client=None, range_theta: Iterable[int] = [0, 90]
    ) -> go.Figure:
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
            for individual in best_simulations["individual"].data:
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
