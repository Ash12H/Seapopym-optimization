"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import cf_xarray
import cf_xarray.units  # noqa: F401
import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr
from seapopym.configuration.no_transport.configuration import KernelParameters
from seapopym.configuration.parameters.parameter_environment import EnvironmentParameter

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
    """
    The structure used to store the observation and compute the difference with the predicted data.

    Warning:
    -------
    Time sampling must be one of : 1D, 1W or 1ME according to pandas resample function.

    """

    name: str
    observation: xr.Dataset
    """The observations units must be convertible to `BIOMASS_UNITS`."""
    observation_type: str = field(default="daily")
    """The type of observation: 'monthly', 'daily', or 'weekly'."""

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
                f"At least one variable is not convertible to {BIOMASS_UNITS}, which is the unit of the predicted "
                "biomass."
            )
            raise ValueError(msg) from e

        if self.observation_type not in ["daily", "monthly", "weekly"]:
            msg = "The observation type must be 'daily', 'monthly', or 'weekly'. Default is 'daily'."
            raise ValueError(msg)
        self.observation = self._helper_resample_data_by_time_type(self.observation)

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

    def _helper_resample_data_by_time_type(self: Observation, data: xr.DataArray) -> xr.DataArray:
        """Resample the data according to the observation type."""
        if self.observation_type == "daily":
            return data.cf.resample(T="1D").mean().cf.dropna("T", how="all")
        if self.observation_type == "monthly":
            return data.cf.resample(T="1ME").mean().cf.dropna("T", how="all")
        if self.observation_type == "weekly":
            return data.cf.resample(T="1W").mean().cf.dropna("T", how="all")

        msg = "The observation type must be 'daily', 'monthly', or 'weekly'. Default is 'daily'."
        raise ValueError(msg)

    def _helper_day_night_apply(
        self: Observation, predicted: xr.Dataset, day_layer: Sequence[int], night_layer: Sequence[int]
    ) -> xr.Dataset:
        """Apply the aggregation and resampling to the predicted data."""
        predicted = predicted.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
        # TODO(Jules): Select the space coordinates -> same as observation.
        predicted = self._helper_resample_data_by_time_type(predicted)

        aggregated_prediction_day = self.aggregate_prediction_by_layer(predicted, day_layer, "day")
        aggregated_prediction_night = self.aggregate_prediction_by_layer(predicted, night_layer, "night")

        return {"day": aggregated_prediction_day, "night": aggregated_prediction_night}

    def mean_square_error(
        self: Observation,
        predicted: xr.Dataset,
        day_layer: Sequence[int],
        night_layer: Sequence[int],
        *,
        centered: bool = False,
        root: bool = False,
        normalized: bool = False,
    ) -> tuple[float | None, float | None]:
        """
        Return the mean square error of the predicted and observed biomass.

        Parameters
        ----------
        predicted : xr.Dataset
            The predicted biomass.
        day_layer : Sequence[int]
            The position of the functional groups during the day.
        night_layer : Sequence[int]
            The position of the functional groups during the night.
        centered : bool
            If True, return the Centered (unbiased) root mean square error (CRMSE).
        root : bool
            If True, the square root of the mean square error is returned.
        normalized : bool
            If True, the mean square error is divided by the standard deviation of the observation.

        """

        def _mse(pred: xr.DataArray, obs: xr.DataArray) -> float:
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
            # WARNING(Jules): What is happening if there are several layers? Should we sum the cost?
            return cost

        cost_day = 0
        cost_night = 0
        aggregated_prediction = self._helper_day_night_apply(predicted, day_layer, night_layer)
        if "day" in self.observation:
            cost_day = _mse(pred=aggregated_prediction["day"], obs=self.observation["day"])
        if "night" in self.observation:
            cost_night = _mse(pred=aggregated_prediction["night"], obs=self.observation["night"])

        return cost_day, cost_night

    def correlation_coefficient(
        self: Observation,
        predicted: xr.Dataset,
        day_layer: Sequence[int],
        night_layer: Sequence[int],
        *,
        corr_dim: str = "time",
    ) -> tuple[float | None, float | None]:
        """Return the correlation coefficient of the predicted and observed biomass."""
        aggregated_prediction = self._helper_day_night_apply(predicted, day_layer, night_layer)
        correlation_day = None
        correlation_night = None
        if "day" in self.observation:
            correlation_day = xr.corr(aggregated_prediction["day"], self.observation["day"], dim=corr_dim)
        if "night" in self.observation:
            correlation_night = xr.corr(aggregated_prediction["night"], self.observation["night"], dim=corr_dim)
        return correlation_day, correlation_night

    def normalized_standard_deviation(
        self: Observation, predicted: xr.Dataset, day_layer: Sequence[int], night_layer: Sequence[int]
    ) -> tuple[float | None, float | None]:
        """Return the normalized standard deviation of the predicted and observed biomass."""
        aggregated_prediction = self._helper_day_night_apply(predicted, day_layer, night_layer)
        normalized_standard_deviation_day = None
        normalized_standard_deviation_night = None
        if "day" in self.observation:
            normalized_standard_deviation_day = aggregated_prediction["day"].std() / self.observation["day"].std()
        if "night" in self.observation:
            normalized_standard_deviation_night = aggregated_prediction["night"].std() / self.observation["night"].std()
        return normalized_standard_deviation_day, normalized_standard_deviation_night

    # TODO(Jules): Add bias
    def bias(self: Observation, predicted: xr.Dataset, day_layer: Sequence[int], night_layer: Sequence[int]) -> None:
        """Return the bias of the predicted and observed biomass."""
        raise NotImplementedError("The bias is not implemented yet.")


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

    functional_groups: Sequence[FunctionalGroupOptimizeNoTransport] | AllGroups
    forcing_parameters: ForcingParameters
    observations: Sequence[Observation]

    def __post_init__(self: GenericCostFunction) -> None:
        """Check that the kwargs are set."""
        if not isinstance(self.functional_groups, AllGroups):
            self.functional_groups = AllGroups(self.functional_groups)

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

    environment_parameters: EnvironmentParameter | None = None
    kernel_parameters: KernelParameters | None = None
    centered_mse: bool = False
    root_mse: bool = True
    normalized_mse: bool = True

    def __post_init__(self: NoTransportCostFunction) -> None:
        """Check that the kwargs are set."""
        super().__post_init__()

    def _cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: Sequence[Observation],
        environment_parameters: EnvironmentParameter | None = None,
        kernel_parameters: KernelParameters | None = None,
    ) -> tuple:
        groups_name = self.functional_groups.functional_groups_name
        filled_args = self.functional_groups.generate_matrix(args)
        day_layers = filled_args[:, NO_TRANSPORT_DAY_LAYER_POS].flatten()
        night_layers = filled_args[:, NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

        fg_parameters = FunctionalGroupGeneratorNoTransport(filled_args, groups_name)

        model = model_generator_no_transport(
            forcing_parameters,
            fg_parameters,
            environment_parameters=environment_parameters,
            kernel_parameters=kernel_parameters,
        )

        model.run()

        predicted_biomass = model.export_biomass()

        return tuple(
            sum(
                obs.mean_square_error(
                    predicted=predicted_biomass,
                    day_layer=day_layers,
                    night_layer=night_layers,
                    centered=self.centered_mse,
                    root=self.root_mse,
                    normalized=self.normalized_mse,
                )
            )
            for obs in observations
        )
