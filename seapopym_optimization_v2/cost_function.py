"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Callable, Iterable, Sequence

import cf_xarray
import cf_xarray.units  # noqa: F401
import numpy as np
import pandas as pd
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr
np.int = int  # Patch temporaire pour compatibilité pygam
from pygam import LinearGAM, s, l
import scipy.sparse
# Patch temporaire : ajoute un attribut `.A` à scipy.sparse matrices
scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
from seapopym.configuration.no_transport.kernel_parameter import KernelParameter
from seapopym.configuration.no_transport.environment_parameter import EnvironmentParameter

from seapopym_optimization.functional_groups import AllGroups, FunctionalGroupOptimizeNoTransport
from seapopym_optimization.wrapper import (
    NO_TRANSPORT_DAY_LAYER_POS,
    NO_TRANSPORT_NIGHT_LAYER_POS,
    FunctionalGroupGeneratorNoTransport,
    model_generator_no_transport,
    FunctionalGroupGeneratorAcidity,
    model_generator_acidity,
)

if TYPE_CHECKING:
    from seapopym.configuration.no_transport.forcing_parameter import ForcingParameter
    from seapopym.configuration.acidity.forcing_parameter import ForcingParameter
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
        log_transform: bool =False,
        eps: float = 1e-6,
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
            if log_transform:
                pred=np.log10(np.maximum(pred,eps)) # avoid log(0)
                obs=np.log10(np.maximum(obs,eps))
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
    forcing_parameters : ForcingParameter
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
    forcing_parameters: ForcingParameter
    observations: Sequence[Observation]

    def __post_init__(self: GenericCostFunction) -> None:
        """Check that the kwargs are set."""
        if not isinstance(self.functional_groups, AllGroups):
            self.functional_groups = AllGroups(self.functional_groups)

    @abstractmethod
    def _cost_function(
        self: GenericCostFunction,
        args: np.ndarray,
        forcing_parameters: ForcingParameter,
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
    forcing_parameters : ForcingParameter
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.

    """

    environment_parameters: EnvironmentParameter | None = None
    kernel_parameters: KernelParameter | None = None
    centered_mse: bool = False
    root_mse: bool = True
    normalized_mse: bool = True
    log_transform_mse: bool =False

    def __post_init__(self: NoTransportCostFunction) -> None:
        """Check that the kwargs are set."""
        super().__post_init__()

    def _cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        forcing_parameters: ForcingParameter,
        observations: Sequence[Observation],
        environment_parameters: EnvironmentParameter | None = None,
        kernel_parameters: KernelParameter | None = None,
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

        predicted_biomass = model.state["biomass"]

        return tuple(
            sum(
                obs.mean_square_error(
                    predicted=predicted_biomass,
                    day_layer=day_layers,
                    night_layer=night_layers,
                    centered=self.centered_mse,
                    root=self.root_mse,
                    normalized=self.normalized_mse,
                    log_transform=self.log_transform_mse
                )
            )
            for obs in observations
        )

@dataclass
class AcidityCostFunction(GenericCostFunction):
    """
    Generator of the cost function for the 'SeapoPym Acidity' model.

    Attributes
    ----------
    functional_groups: Sequence[FunctionalGroupOptimizeAcidity]
        The list of functional groups.
    forcing_parameters : ForcingParameter
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.

    """

    environment_parameters: EnvironmentParameter | None = None
    kernel_parameters: KernelParameter | None = None
    centered_mse: bool = False
    root_mse: bool = True
    normalized_mse: bool = True
    log_transform_mse: bool =False

    def __post_init__(self: AcidityCostFunction) -> None:
        """Check that the kwargs are set."""
        super().__post_init__()

    def _cost_function(
        self: AcidityCostFunction,
        args: np.ndarray,
        forcing_parameters: acidity.ForcingParameter,
        observations: Sequence[Observation],
        environment_parameters: EnvironmentParameter | None = None,
        kernel_parameters: KernelParameter | None = None,
    ) -> tuple:
        groups_name = self.functional_groups.functional_groups_name
        filled_args = self.functional_groups.generate_matrix(args)
        day_layers = filled_args[:, NO_TRANSPORT_DAY_LAYER_POS].flatten()
        night_layers = filled_args[:, NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

        fg_parameters = FunctionalGroupGeneratorAcidity(filled_args, groups_name)

        model = model_generator_acidity(
            forcing_parameters,
            fg_parameters,
            environment_parameters=environment_parameters,
            kernel_parameters=kernel_parameters,
        )

        model.run()

        predicted_biomass = model.state["biomass"]

        return tuple(
            sum(
                obs.mean_square_error(
                    predicted=predicted_biomass,
                    day_layer=day_layers,
                    night_layer=night_layers,
                    centered=self.centered_mse,
                    root=self.root_mse,
                    normalized=self.normalized_mse,
                    log_transform=self.log_transform_mse
                )
            )
            for obs in observations
        )
    
@dataclass
class GAMPteropodCostFunction(GenericCostFunction):
    """
    Generator of the cost function for the 'SeapoPym Acidity' model.
    Using GAM decomposition (trend/seasonality/residuals)

    Attributes
    ----------
    functional_groups: Sequence[FunctionalGroupOptimizeAcidity]
        The list of functional groups.
    forcing_parameters : ForcingParameter
        Forcing parameters.
    observations : Sequence[Observation]
        Observations.

    Optional
    --------
    weights : list of float
        relative weights in the cost function assigned to :
        0 the trend
        1 the seasonal component
        default is [0.5,0.5]

    WARNING : in this class, data is automaticaly log10 transfrom    

    """

    environment_parameters: EnvironmentParameter | None = None
    kernel_parameters: KernelParameter | None = None
    weights: list[float] = field(default_factory=lambda: [0.5,0.5])

    def __post_init__(self: GAMPteropodCostFunction) -> None:
        """Check that the kwargs are set."""
        super().__post_init__()

    def decompose_GAM(self,data,variable,eps: float = 1e-6):
        """Decompose time series using GAM model into trend and seasonality, 
        all the calculations are in the log10 base
        
        Parameters:
             data (dataframe): must contain 'time' and the target variable to decompose
             variable (str) : name of the variable in the model
             eps (float): small value to avoid log(0)
        Returns:
            (trend_df,season_df):DataFrame with 'time' and 'biomass' columns
        
        """
        data=data.copy()
        data[variable]=np.log10(np.maximum(data[variable],eps)) # log10 transformation, epsilon to avoid log(0) 

        data = data.dropna().reset_index(drop=True)
        data['time_float'] = (data['time'] - data['time'].min()).dt.total_seconds() / (3600 * 24)

        data['month'] = data['time'].dt.month
        data['month_sin'] = np.sin(2 * np.pi * (data['month'] - 1) / 12)
        data['month_cos'] = np.cos(2 * np.pi * (data['month'] - 1) / 12)

        X = data[['time_float', 'month_sin', 'month_cos']].values
        y = data[variable].values
       
        # For the estimation of the long-term trend, we use a spline term with n_splines=80.
        # This controls the flexibility of the spline fit over time.
        # - A higher n_splines allows the model to capture more rapid changes (but also more noise).
        # - A lower n_splines results in a smoother trend that captures only large-scale variations.
        gam = LinearGAM(s(0, n_splines=80) + l(1) + l(2), fit_intercept=False).fit(X, y)

        trend = gam.partial_dependence(term=0, X=X)
        season = gam.partial_dependence(term=1, X=X) + gam.partial_dependence(term=2, X=X)

        trend_df=pd.DataFrame({
            "time": data["time"].values,
            "biomass": trend
        })
        season_df=pd.DataFrame({
            "time": data["time"].values,
            "biomass": season
        })

        return trend_df,season_df

    def _cost_function(
        self: GAMPteropodCostFunction,
        args: np.ndarray,
        forcing_parameters: acidity.ForcingParameter,
        observations: Sequence[Observation],
        environment_parameters: EnvironmentParameter | None = None,
        kernel_parameters: KernelParameter | None = None,
    ) -> tuple:
        groups_name = self.functional_groups.functional_groups_name
        filled_args = self.functional_groups.generate_matrix(args)
        day_layers = filled_args[:, NO_TRANSPORT_DAY_LAYER_POS].flatten()
        night_layers = filled_args[:, NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

        fg_parameters = FunctionalGroupGeneratorAcidity(filled_args, groups_name)

        model = model_generator_acidity(
            forcing_parameters,
            fg_parameters,
            environment_parameters=environment_parameters,
            kernel_parameters=kernel_parameters,
        )

        model.run()

        predicted_biomass = model.state["biomass"]

        def RMSE(obs,pred):
            """compute squared, normalised RMSE"""
            # align in time obs and pred
            df=pd.merge(obs,pred,on="time",how="inner",suffixes=("_obs","_pred"))
            # compute RMSE
            cost=float(((df["biomass_obs"]-df["biomass_pred"])**2).mean())
            cost=np.sqrt(cost)
            cost/=float(df["biomass_obs"].std())
            return cost

        cost=[]
        for obs in observations:
            predicted = obs._helper_resample_data_by_time_type(predicted_biomass)
            predicted = predicted.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
            obs.observation = obs.observation.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
            obs_df=pd.DataFrame({
                "time": obs.observation["time"].values,
                "day": obs.observation.to_array().squeeze().values
            })
            pred_df=pd.DataFrame({
                "time": predicted["time"].values[3:],
                "biomass": predicted.squeeze().values[3:] # [3:] to rm the first 3 months (let the model stabilise)
            })
            obs_trend, obs_season = self.decompose_GAM(obs_df, "day")
            pred_trend, pred_season = self.decompose_GAM(pred_df, "biomass")

            rmse_trend = self.weights[0]*RMSE(obs_trend, pred_trend)
            rmse_season = self.weights[1]*RMSE(obs_season, pred_season)

            cost.append(rmse_trend + rmse_season)
                
        return tuple(cost)
