"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Callable, Iterable, Sequence

import cf_xarray
import cf_xarray.units
import numpy as np
import pint
import pint_xarray
import xarray as xr
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.parameter import ForcingParameters, NoTransportParameters
from seapopym.model.no_transport_model import NoTransportModel

from seapopym_optimization.wrapper import FunctionalGroupGeneratorNoTransport

BIOMASS_UNITS = "kg/m2"


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
    nb_parameters : int
        Number of parameters by functional group in the simulation.
    nb_functional_groups : int
        Number of functional groups in the simulation.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : ...
        Observations.
    fixed_parameters : np.ndarray
        Fixed parameters.

    Notes
    -----
    This class is used to create a generic cost function that can be used to optimize the parameters of the SeapoPym
    model. The cost function must be rewritten in the child class following the steps below:
    #TODO(Jules): Add the steps to follow to create a new cost function.

    """

    nb_parameters: int
    nb_functional_groups: int
    forcing_parameters: ForcingParameters
    observations: Sequence[Observation]
    fixed_parameters: np.ndarray | None = None

    def __post_init__(self: GenericCostFunction) -> None:
        """Check validity of the class."""
        if self.fixed_parameters is None:
            self.fixed_parameters = np.full((self.nb_functional_groups, self.nb_parameters), np.nan, dtype=float)
        else:
            self.fixed_parameters = np.asarray(self.fixed_parameters, dtype=float)

        if self.fixed_parameters.shape != (self.nb_functional_groups, self.nb_parameters):
            msg = f"Fixed parameters must have the shape ({self.nb_functional_groups}, {self.nb_parameters})"
            raise ValueError(msg)

    @abstractmethod
    def cost_function(
        self: GenericCostFunction,
        args: np.ndarray,
        fixed_parameters: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: ...,
    ) -> tuple:
        """
        Calculate the cost of the simulation.

        This function must be rewritten in the child class.
        """

    def generate(self: GenericCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(
            self.cost_function,
            fixed_parameters=self.fixed_parameters,
            forcing_parameters=self.forcing_parameters,
            observations=self.observations,
        )


@dataclass
class NoTransportCostFunction(GenericCostFunction):
    """
    Generator of the cost function for the 'SeapoPym No Transport' model. Consider all stations at once without
    weights.

    Attributes
    ----------
    nb_parameters : int
        Number of parameters by functional group in the simulation.
    nb_functional_groups : int
        Number of functional groups in the simulation.
    groups_name : list[str]
        List of the functional groups name.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.
    fixed_parameters : np.ndarray
        Fixed parameters. The parameters order is : tr_max, tr_rate, inv_lambda_max, inv_lambda_rate, day_layer,
        night_layer, energy_transfert.

    """

    groups_name: list[str] = None

    def __post_init__(self: NoTransportCostFunction) -> None:
        """Check validity of the class."""
        super().__post_init__()

        if self.groups_name is None:
            self.groups_name = [f"FG_{i}" for i in range(self.nb_functional_groups)]

    def fill_args(self: NoTransportCostFunction, args: np.ndarray, fixed_parameters: np.ndarray) -> np.ndarray:
        """
        Fill the fixed parameters in the args. Used to get all the parameters needed for the simulation.

        Parameters
        ----------
        args : np.ndarray
            Parameters to optimize. This array must be flattened.
        fixed_parameters : np.ndarray
            Fixed parameters. This array must have the shape (nb_functional_groups, nb_parameters).

        Returns
        -------
        np.ndarray
            An array that contains all the parameters needed for the simulation. The shape is the same as the
            fixed_parameters array.

        """
        initial_shape = fixed_parameters.shape
        args = np.asarray(args, dtype=float)
        args_flat = fixed_parameters.flatten()
        args_flat[np.isnan(args_flat)] = args
        return args_flat.reshape(initial_shape)

    def cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        fixed_parameters: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: Sequence[Observation],
        groups_name: list[str],
        **kwargs: dict,
    ) -> tuple:
        args = self.fill_args(args, fixed_parameters)
        fg_parameters = FunctionalGroupGeneratorNoTransport(args, groups_name)
        day_layers = args[:, 4].flatten()
        night_layers = args[:, 5].flatten()

        model = NoTransportModel(
            configuration=NoTransportConfiguration(
                parameters=NoTransportParameters(
                    forcing_parameters=forcing_parameters,
                    functional_groups_parameters=fg_parameters.generate(),
                    **kwargs,
                )
            )
        )

        model.run()

        predicted_biomass = model.export_biomass()

        return tuple(
            obs.mean_square_error(predicted=predicted_biomass, day_layer=day_layers, night_layer=night_layers)
            for obs in observations
        )

    def generate(self: NoTransportCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(
            self._cost_function,
            fixed_parameters=self.fixed_parameters,
            forcing_parameters=self.forcing_parameters,
            observations=self.observations,
            groups_name=self.groups_name,
        )
