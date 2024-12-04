"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import cf_xarray
import cf_xarray.units  # noqa: F401
import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

from seapopym_optimization.functional_groups import AllGroups, FunctionalGroupOptimizeNoTransport
from seapopym_optimization.wrapper import (
    NO_TRANSPORT_DAY_LAYER_POS,
    NO_TRANSPORT_NIGHT_LAYER_POS,
    FunctionalGroupGeneratorNoTransport,
    model_generator_no_transport,
)

if TYPE_CHECKING:
    from seapopym.configuration.no_transport.parameter import ForcingParameters

BIOMASS_UNITS = "g/m2"
MAXIMUM_INIT_TRY = 1000


@dataclass
class Observation:
    """The structure used to store the observation and compute the difference with the predicted data."""

    observation: xr.Dataset
    """The observations units must be convertible to `BIOMASS_UNITS`."""

    def __post_init__(self: Observation) -> None:
        """Check that the observation data is complient with the format of the predicted biomass."""
        for coord in ["T", "X", "Y", "Z"]:
            if coord not in self.observation.cf.coords:
                msg = f"Coordinate {coord} must be in the observation Dataset."
                raise ValueError(msg)

        try:
            self.observation.pint.quantify().pint.dequantify()
        except Exception as e:
            msg = (
                "You must specify units for each variable and axis for each coordinate. Refer to CF_XARRAY"
                "documentation for coordinates and PINT for units."
            )
            raise ValueError(msg) from e

        try:
            for variable in self.observation:
                self.observation[variable] = (
                    self.observation[variable].pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
                )
        except Exception as e:
            msg = (
                f"At least one variable is not convertible to {BIOMASS_UNITS}, which is the unit of the predicted ",
                "biomass.",
            )
            raise ValueError(msg) from e

    def aggregate_prediction_by_layer(
        self: Observation, predicted: xr.DataArray, position: Sequence[int], name: str
    ) -> xr.DataArray:
        """
        The `predicted` DataArray is aggregated by layer depending on the `position` of the functional groups during
        night/day.
        """
        z_coord = self.observation.cf["Z"].name
        final_aggregated = []

        for layer_position in self.observation.cf["Z"].data:
            functional_group = predicted["functional_group"].data[(np.asarray(position) == layer_position)]
            aggregated_predicted = predicted.sel(functional_group=functional_group).sum("functional_group")
            aggregated_predicted = aggregated_predicted.expand_dims({z_coord: [layer_position]})
            final_aggregated.append(aggregated_predicted)

        return xr.concat(final_aggregated, dim=z_coord).rename(name)

    def mean_square_error(
        self: Observation, predicted: xr.Dataset, night_layer: Sequence[int], day_layer: Sequence[int]
    ) -> None:
        """Return the mean square error of the predicted and observed biomass."""

        def _mse(pred: xr.DataArray, obs: xr.DataArray) -> float:
            """Mean square error applied to xr.DataArray."""
            cost = float(((obs - pred) ** 2).mean())
            if not np.isfinite(cost):
                msg = (
                    "Nan value in cost function. The observation cannot be compared to the prediction. Verify that "
                    "coordinates are fitting both in space and time."
                )
                raise ValueError(msg)
            return cost

        predicted = predicted.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()

        cost = 0
        if "day" in self.observation:
            aggregated_prediction_day = self.aggregate_prediction_by_layer(predicted, day_layer, "day")
            cost += _mse(pred=aggregated_prediction_day, obs=self.observation["day"])
        if "night" in self.observation:
            aggregated_prediction_night = self.aggregate_prediction_by_layer(predicted, night_layer, "night")
            cost += _mse(pred=aggregated_prediction_night, obs=self.observation["night"])

        return cost


@dataclass
class GenericCostFunction(ABC):
    """
    Generic cost function class.

    Parameters
    ----------
    functional_groups: Sequence[GenericFunctionalGroupOptimize]
        ...
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : ...
        Observations.

    Notes
    -----
    This class is used to create a generic cost function that can be used to optimize the parameters of the SeapoPym
    model. The cost function must be rewritten in the child class following the steps below:
    #TODO(Jules): Add the steps to follow to create a new cost function.

    """

    functional_groups: Sequence[FunctionalGroupOptimizeNoTransport]
    forcing_parameters: ForcingParameters
    observations: Sequence[Observation]

    @property
    @abstractmethod
    def parameters_name(self: GenericCostFunction) -> Sequence[str]:
        """Return the ordered list of parameters name."""

    @abstractmethod
    def _cost_function(
        self: GenericCostFunction,
        args: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: Sequence[Observation],
        **kwargs: dict,
    ) -> tuple:
        """
        Calculate the cost of the simulation.

        This function must be rewritten in the child class.
        """

    def generate(self: GenericCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(
            self._cost_function,
            forcing_parameters=self.forcing_parameters,
            observations=self.observations,
        )


@dataclass
class NoTransportCostFunction(GenericCostFunction):
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

    kwargs: dict | None = None
    # TODO(Jules): Replace kwargs by the NoTransport configuration structure -> Env and Kernel

    def __post_init__(self: NoTransportCostFunction) -> None:
        """Check that the kwargs are set."""
        if self.kwargs is None:
            self.kwargs = {}

    @property
    def parameters_name(self: NoTransportCostFunction) -> Sequence[str]:
        """Return the ordered list of parameters name."""
        # NOTE(Jules): AllGroups is instantiated 2 times. This is not optimal.
        return AllGroups(self.functional_groups).get_all_parameters_names_ordered()

    def _cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: Sequence[Observation],
        **kwargs: dict,
    ) -> tuple:
        all_groups = AllGroups(self.functional_groups)
        filled_args = all_groups.generate_matrix(args)
        groups_name = all_groups.groups_name
        day_layers = filled_args[:, NO_TRANSPORT_DAY_LAYER_POS].flatten()
        night_layers = filled_args[:, NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

        fg_parameters = FunctionalGroupGeneratorNoTransport(filled_args, groups_name)

        model = model_generator_no_transport(forcing_parameters, fg_parameters, **kwargs)

        model.run()

        predicted_biomass = model.export_biomass().load()

        return tuple(
            obs.mean_square_error(predicted=predicted_biomass, day_layer=day_layers, night_layer=night_layers)
            for obs in observations
        )
