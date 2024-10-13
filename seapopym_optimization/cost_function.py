"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

import random
from abc import ABC, abstractmethod, abstractproperty
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
class Parameter:
    """
    The definition of a parameter to optimize.

    Parameters
    ----------
    name: str
        ...
    lower_bound: float
        ...
    upper_bound: float
        ...
    init_method: Callable[[float, float], float], optional
        The method used to get the initial value of a parameter. Default is a random uniform distribution that exclude
        the bounds values.

    """

    name: str
    lower_bound: float
    upper_bound: float
    init_method: Callable[[float, float], float] = None

    def __post_init__(self: Parameter) -> None:
        if self.lower_bound >= self.upper_bound:
            msg = f"Lower bounds ({self.lower_bound}) must be <= to upper bound ({self.upper_bound})."
            raise ValueError(msg)

        if self.init_method is None:

            def random_exclusive(lower: float, upper: float) -> float:
                count = 0
                while count < MAXIMUM_INIT_TRY:
                    value = random.uniform(lower, upper)
                    if value not in (lower, upper):
                        return value
                    count += 1
                msg = f"Random parameter initialization reach maximum try for parameter {self.name}"
                raise ValueError(msg)

            self.init_method = random_exclusive


@dataclass
class GenericFunctionalGroupOptimize(ABC):
    """The Generic structure used to store the parameters of a functional group as used in SeapoPym."""

    name: str

    @property
    @abstractmethod
    def parameters(self: GenericFunctionalGroupOptimize) -> tuple:
        """
        Return the parameters representing the functional group.

        Warning:
        -------
        Order of declaration is the same as in the cost_function.

        """

    @abstractmethod
    def as_tuple(self: GenericFunctionalGroupOptimize) -> Sequence[float]:
        """
        Return a tuple that contains all the functional group parameters (except name) as float values. When value is
        not set, return np.NAN.
        """

    @abstractmethod
    def get_parameters_to_optimize() -> Sequence[Parameter]:
        """Return the parameters to optimize as a sequence of `Parameter`."""


@dataclass
class FunctionalGroupOptimizeNoTransport(GenericFunctionalGroupOptimize):
    """The parameters of a functional group as they are defined in the SeapoPym NoTransport model."""

    tr_max: float | Parameter
    tr_rate: float | Parameter
    inv_lambda_max: float | Parameter
    inv_lambda_rate: float | Parameter
    day_layer: float | Parameter
    night_layer: float | Parameter
    energy_coefficient: float | Parameter

    @property
    def parameters(self: FunctionalGroupOptimizeNoTransport) -> tuple:
        return (
            self.tr_max,
            self.tr_rate,
            self.inv_lambda_max,
            self.inv_lambda_rate,
            self.day_layer,
            self.night_layer,
            self.energy_coefficient,
        )

    def as_tuple(self: FunctionalGroupOptimizeNoTransport) -> tuple:
        return tuple(np.nan if isinstance(param, Parameter) else param for param in self.parameters)

    def get_parameters_to_optimize(self: FunctionalGroupOptimizeNoTransport) -> Sequence[Parameter]:
        return tuple(param for param in self.parameters if isinstance(param, Parameter))


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

    functional_groups: Sequence[GenericFunctionalGroupOptimize]
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
        groups_name: Sequence[str],
        fixed_parameters: np.ndarray,
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
        groups_name = tuple(fg.name for fg in self.functional_groups)
        fixed_parameters = np.asarray(tuple(fg.as_tuple() for fg in self.functional_groups))
        return partial(
            self._cost_function,
            groups_name=groups_name,
            fixed_parameters=fixed_parameters,
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

    NO_TRANSPORT_DAY_LAYER_POS = 4
    NO_TRANSPORT_NIGHT_LAYER_POS = 5

    @property
    def parameters_name(self: NoTransportCostFunction) -> Sequence[str]:
        names = []
        for fg in self.functional_groups:
            names += [param.name for param in fg.get_parameters_to_optimize()]
        return names

    def _fill_args(self: NoTransportCostFunction, args: np.ndarray, fixed_parameters: np.ndarray) -> np.ndarray:
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

    def _cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        groups_name: Sequence[str],
        fixed_parameters: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: Sequence[Observation],
        **kwargs: dict,
    ) -> tuple:
        args = self._fill_args(args, fixed_parameters)
        fg_parameters = FunctionalGroupGeneratorNoTransport(args, groups_name)
        day_layers = args[:, self.NO_TRANSPORT_DAY_LAYER_POS].flatten()
        night_layers = args[:, self.NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

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
