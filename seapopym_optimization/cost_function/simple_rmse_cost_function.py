"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels

from seapopym_optimization.cost_function.base_cost_function import AbstractCostFunction
from seapopym_optimization.cost_function.observations import DayCycle

if TYPE_CHECKING:
    from collections.abc import Sequence

    from seapopym_optimization.cost_function.observations import TimeSeriesObservation


def aggregate_biomass_by_layer(
    data: xr.DataArray,
    position: Sequence[int],
    name: str,
    layer_coordinates: Sequence[int],
    layer_coordinates_name: str = "layer",
) -> xr.DataArray:
    """Aggregate biomass data by layer coordinates."""
    layer_coord = xr.DataArray(
        np.asarray(position),
        dims=[CoordinatesLabels.functional_group],
        coords={CoordinatesLabels.functional_group: data[CoordinatesLabels.functional_group].data},
        name=layer_coordinates_name,
    )
    return (
        data.assign_coords({layer_coordinates_name: layer_coord})
        .groupby(layer_coordinates_name)
        .sum(dim=CoordinatesLabels.functional_group)
        .reindex({layer_coordinates_name: layer_coordinates})
        .fillna(0)
        .rename(name)
    )
    # # ---------------------------------------------------------------------------------------------------------------- #
    # # NOTE(Jules): Old implementation, kept for reference.
    # final_aggregated = []
    # for layer_position in layer_coordinates:
    #     functional_group = data[CoordinatesLabels.functional_group].data[(np.asarray(position) == layer_position)]
    #     aggregated_predicted = data.sel(functional_group=functional_group).sum("functional_group")
    #     aggregated_predicted = aggregated_predicted.expand_dims({layer_coordinates_name: [layer_position]})
    #     final_aggregated.append(aggregated_predicted)
    # return xr.concat(final_aggregated, dim=layer_coordinates_name).rename(name)
    # # ---------------------------------------------------------------------------------------------------------------- #


def root_mean_square_error(
    pred: xr.DataArray,
    obs: xr.DataArray,
    *,
    root: bool,
    centered: bool,
    normalized: bool,
) -> float:
    """Mean square error applied to xr.DataArray."""
    if centered:
        cost = float(((pred - pred.mean()) - (obs - obs.mean())).mean() ** 2)
    else:
        cost = float(((obs - pred) ** 2).mean())
    if root:
        cost = np.sqrt(cost)
    if normalized:
        cost /= float(obs.std())
    if not np.isfinite(cost):
        msg = (
            "Nan value in cost function. The observation cannot be compared to the prediction. Verify that "
            "coordinates are fitting both in space and time."
        )
        raise ValueError(msg)
    return cost


@dataclass
class SimpleRootMeanSquareErrorCostFunction(AbstractCostFunction):
    """
    Generator of the cost function for the 'SeapoPym No Transport' model.

    Attributes
    ----------
    functional_groups: Sequence[FunctionalGroupOptimizeNoTransport]
        The list of functional groups.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.

    """

    observations: Sequence[TimeSeriesObservation]
    root_mse: bool = True
    centered_mse: bool = False
    normalized_mse: bool = False

    def _cost_function(self: SimpleRootMeanSquareErrorCostFunction, args: np.ndarray) -> tuple:
        model = self.model_generator.generate(
            functional_group_names=self.functional_groups.functional_groups_name(),
            functional_group_parameters=self.functional_groups.generate(args),
        )

        model.run()

        predicted_biomass = model.state[ForcingLabels.biomass]

        biomass_day = aggregate_biomass_by_layer(
            data=predicted_biomass,
            position=model.state[ConfigurationLabels.day_layer].data,
            name=DayCycle.DAY,
            layer_coordinates=model.state.cf[CoordinatesLabels.Z].data,  # TODO(Jules): layer_coordinates ?
        )
        biomass_night = aggregate_biomass_by_layer(
            data=predicted_biomass,
            position=model.state[ConfigurationLabels.night_layer].data,
            name=DayCycle.NIGHT,
            layer_coordinates=model.state.cf[CoordinatesLabels.Z].data,
        )

        return tuple(
            root_mean_square_error(
                pred=biomass_day if obs.observation_type == DayCycle.DAY else biomass_night,
                obs=obs.observation,
                root=self.root_mse,
                centered=self.centered_mse,
                normalized=self.normalized_mse,
            )
            for obs in self.observations
        )
