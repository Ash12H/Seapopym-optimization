from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cf_xarray
import numpy as np
import pandas as pd
from pygam import LinearGAM, l, s
from seapopym.standard.labels import ConfigurationLabels, CoordinatesLabels, ForcingLabels
from statsmodels.tsa import seasonal

from seapopym_optimization.cost_function.base_observation import DayCycle
from seapopym_optimization.cost_function.simple_rmse_cost_function import (
    SimpleRootMeanSquareErrorCostFunction,
    TimeSeriesObservation,
    aggregate_biomass_by_layer,
    root_mean_square_error,
)

if TYPE_CHECKING:
    from numbers import Number


# @dataclass
# class GAMPteropodCostFunction(GenericCostFunction):
#     """
#     Generator of the cost function for the 'SeapoPym Acidity' model.
#     Using GAM decomposition (trend/seasonality/residuals).

#     Attributes
#     ----------
#     functional_groups: Sequence[FunctionalGroupOptimizeAcidity]
#         The list of functional groups.
#     forcing_parameters : ForcingParameter
#         Forcing parameters.
#     observations : Sequence[Observation]
#         Observations.

#     Optional
#     --------
#     weights : list of float
#         relative weights in the cost function assigned to :
#         0 the trend
#         1 the seasonal component
#         default is [0.5,0.5]

#     WARNING : in this class, data is automaticaly log10 transfrom

#     """

#     environment_parameters: EnvironmentParameter | None = None
#     kernel_parameters: KernelParameter | None = None
#     weights: list[float] = field(default_factory=lambda: [0.5, 0.5])

#     def __post_init__(self: GAMPteropodCostFunction) -> None:
#         """Check that the kwargs are set."""
#         super().__post_init__()

#     def decompose_GAM(self, data, variable):
#         """
#         Decompose time series using GAM model into trend and seasonality,
#         all the calculations are in the log10 base.

#         Parameters:
#             data (dataframe): must contain 'time' and the target variable to decompose
#             variable (str) : name of the variable in the model

#         Returns:
#             (trend_df,season_df):DataFrame with 'time' and 'biomass' columns

#         """
#         data = data.copy()
#         data[variable] = np.log10(
#             np.maximum(data[variable], np.finfo(float).eps)
#         )  # log10 transformation, epsilon to avoid log(0)

#         data = data.dropna().reset_index(drop=True)
#         data["time_float"] = (data["time"] - data["time"].min()).dt.total_seconds() / (3600 * 24)

#         data["month"] = data["time"].dt.month
#         data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12)
#         data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12)

#         X = data[["time_float", "month_sin", "month_cos"]].values
#         y = data[variable].values

#         # For the estimation of the long-term trend, we use a spline term with n_splines=80.
#         # This controls the flexibility of the spline fit over time.
#         # - A higher n_splines allows the model to capture more rapid changes (but also more noise).
#         # - A lower n_splines results in a smoother trend that captures only large-scale variations.
#         gam = LinearGAM(s(0, n_splines=80) + l(1) + l(2), fit_intercept=False).fit(X, y)

#         trend = gam.partial_dependence(term=0, X=X)
#         season = gam.partial_dependence(term=1, X=X) + gam.partial_dependence(term=2, X=X)

#         trend_df = pd.DataFrame({"time": data["time"].values, "biomass": trend})
#         season_df = pd.DataFrame({"time": data["time"].values, "biomass": season})

#         return trend_df, season_df

#     def RMSE(self, obs, pred):
#         """compute squared, normalised RMSE"""
#         # align in time obs and pred
#         df = pd.merge(obs, pred, on="time", how="inner", suffixes=("_obs", "_pred"))
#         # compute RMSE
#         cost = float(((df["biomass_obs"] - df["biomass_pred"]) ** 2).mean())
#         cost = np.sqrt(cost)
#         cost /= float(df["biomass_obs"].std())
#         return cost

#     def _cost_function(
#         self: GAMPteropodCostFunction,
#         args: np.ndarray,
#         forcing_parameters: ForcingParameter,
#         observations: Sequence[Observation],
#         environment_parameters: EnvironmentParameter | None = None,
#         kernel_parameters: KernelParameter | None = None,
#     ) -> tuple:
#         groups_name = self.functional_groups.functional_groups_name
#         filled_args = self.functional_groups.generate_matrix(args)
#         day_layers = filled_args[:, NO_TRANSPORT_DAY_LAYER_POS].flatten()
#         night_layers = filled_args[:, NO_TRANSPORT_NIGHT_LAYER_POS].flatten()

#         fg_parameters = FunctionalGroupGeneratorNoTransport(filled_args, groups_name)

#         model = model_generator_no_transport(
#             forcing_parameters,
#             fg_parameters,
#             environment_parameters=environment_parameters,
#             kernel_parameters=kernel_parameters,
#         )

#         model.run()

#         predicted_biomass = model.state["biomass"]

#         cost = []
#         for obs in observations:
#             predicted = obs._helper_resample_data_by_time_type(predicted_biomass)
#             predicted = predicted.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
#             obs.observation = obs.observation.pint.quantify().pint.to(BIOMASS_UNITS).pint.dequantify()
#             obs_df = pd.DataFrame(
#                 {"time": obs.observation["time"].values, "day": obs.observation.to_array().squeeze().values}
#             )
#             pred_df = pd.DataFrame(
#                 {
#                     "time": predicted["time"].values[3:],
#                     "biomass": predicted.squeeze().values[
#                         3:
#                     ],  # [3:] to rm the first 3 months (let the model stabilise)
#                 }
#             )
#             obs_trend, obs_season = self.decompose_GAM(obs_df, "day")
#             pred_trend, pred_season = self.decompose_GAM(pred_df, "biomass")

#             rmse_trend = self.weights[0] * self.RMSE(obs_trend, pred_trend)
#             rmse_season = self.weights[1] * self.RMSE(obs_season, pred_season)

#             cost.append(rmse_trend + rmse_season)

#         return tuple(cost)


# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class SeasonalObservation(TimeSeriesObservation, ABC):
    """
    SeasonalObservation is an abstract class that represents a seasonal observation of a time series.
    It contains the trend, seasonal, and residuals components of the time series.
    """

    trend: pd.Series = None
    seasonal: pd.Series | pd.DataFrame = None
    residuals: pd.Series = None

    @classmethod
    @abstractmethod
    def from_timeseries_observation(cls, observation: TimeSeriesObservation) -> SeasonalObservation:
        """Create a SeasonalObservation from a TimeSeriesObservation."""


@dataclass(kw_only=True)
class GAMSeasonalObservation(SeasonalObservation):
    """
    GAMSeasonalObservation is a SeasonalObservation that uses GAM decomposition.
    It contains the trend, seasonal, and residuals components of the time series.

    Attributes.
    ----------
    trend : pd.Series
        The trend component of the time series.
    seasonal : pd.Series | pd.DataFrame
        The seasonal component of the time series.
    residuals : pd.Series
        The residuals component of the time series.
    """

    layer_weights: Sequence[Number] | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.layer_weights is None:
            weights = [
                self.observation.cf.sel({CoordinatesLabels.Z: layer}).notnull().sum()
                for layer in self.observation.cf[CoordinatesLabels.Z].data
            ]
            self.layer_weights = np.asarray(weights) / np.sum(weights)

    @classmethod
    def from_timeseries_observation(
        cls: GAMSeasonalObservation,
        observation: TimeSeriesObservation,
        n_splines: int = 20,
        **kwargs: dict,
    ) -> GAMSeasonalObservation:
        """
        Create a GAMSeasonalObservation from a TimeSeriesObservation.

        Parameters
        ----------
        observation : TimeSeriesObservation
            The observation to convert.
        n_splines : int, optional
            Number of splines to use for the trend component, by default 20.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the GAM decomposition function.

        Returns
        -------
        GAMSeasonalObservation
            The converted observation.

        """
        result = decompose_gam(
            time=observation.observation.cf.indexes["T"],
            data=observation.observation.squeeze().data,
            n_splines=n_splines,
            **kwargs,
        )

        return cls(
            name=observation.name,
            observation=observation.observation,
            observation_type=observation.observation_type,
            observation_interval=observation.observation_interval,
            trend=result.trend,
            seasonal=result.seasonal,
            residuals=result.resid,
        )


def _patch_pygam() -> None:
    """Since PyGam isn't compatible with python 3.13, we patch it to work with the latest version."""
    np.int = int  # noqa: NPY001

    import scipy.sparse

    scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())


def decompose_gam(
    time: pd.DatetimeIndex,
    data: Sequence[Number],
    n_splines: int = 20,
    *,
    fit_intercept: bool = False,
    seasonal_cycle_length: Number = 365.25,
    **kwargs: dict,
) -> pd.DataFrame:
    """
    Decompose time series using GAM model into trend and seasonality, all the calculations are in the log10 base.

    Parameters
    ----------
    time : pd.DatetimeIndex
        Time index for the data.
    data : Sequence[Number]
        Sequence of data values to decompose.
    n_splines : int, optional
        Number of splines to use for the trend component, by default 20.
    fit_intercept : bool, optional
        Whether to fit an intercept in the GAM model, by default False.
    seasonal_cycle_length : Number, optional
        Length of the seasonal cycle in days, by default 365.25 (for 1 year seasonality).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the LinearGAM constructor.

    Returns
    -------
        pd.DataFrame: DataFrame with 'time', 'trend', 'season', and 'residuals' columns

    """
    _patch_pygam()

    data = (
        pd.DataFrame({"time": time, "data": data})
        .set_index("time")
        .resample("D")
        .mean()
        .interpolate("linear")
        .reset_index()
    )

    data["day_since_start"] = np.cumsum(np.ones_like(data["time"], dtype=int))
    data["sin_doy"] = np.sin(2 * np.pi * data["day_since_start"] / seasonal_cycle_length)
    data["cos_doy"] = np.cos(2 * np.pi * data["day_since_start"] / seasonal_cycle_length)

    x = data[["day_since_start", "sin_doy", "cos_doy"]].to_numpy()
    y = data["data"].to_numpy()
    gam = LinearGAM(s(0, n_splines=n_splines) + l(1) + l(2), fit_intercept=fit_intercept, **kwargs).fit(x, y)
    trend = gam.partial_dependence(term=0, X=x)
    season = gam.partial_dependence(term=1, X=x) + gam.partial_dependence(term=2, X=x)
    residuals = y - trend - season
    return pd.DataFrame({"time": data["time"], "trend": trend, "seasonal": season, "resid": residuals})


def decompose_season_trend_loess(
    time: pd.DatetimeIndex, data: Sequence[Number], periods: Sequence[int] | int = 365, **kwargs: dict
) -> pd.DataFrame:
    """
    Decompose time series using STL or MSTL decomposition. If the `periods` parameter is a single integer,
    it uses STL decomposition; if it is a sequence of integers, it uses MSTL decomposition.

    Parameters.
    ----------
    time : pd.DatetimeIndex
        Time index for the data.
    data : Sequence[Number]
        Sequence of data values to decompose.
    periods : Sequence[int] | int, optional
        Periods for the seasonal decomposition, can be a single integer or a sequence of integers,
        by default 365 (for 1 year seasonality).
    **kwargs : dict, optional
        Additional keyword arguments to pass to the STL or MSTL constructor.

    Returns
    -------
        pd.DataFrame: DataFrame with 'trend', 'seasonal', and 'resid' columns

    """
    if isinstance(periods, Sequence) and len(periods) == 1:
        periods = periods[0]

    if isinstance(periods, int):
        result = seasonal.STL(
            pd.DataFrame({"time": time, "data": data}).resample("1D").mean().interpolate("linear"),
            periods=periods,
            **kwargs,
        )
        return pd.DataFrame([result.trend, result.seasonal, result.resid])

    if isinstance(periods, Sequence):
        result = seasonal.MSTL(
            pd.DataFrame({"time": time, "data": data}).resample("1D").mean().interpolate("linear"),
            periods=periods,
            **kwargs,
        )
        return pd.DataFrame([result.trend, result.resid]).T.merge(result.seasonal, on="time")

    msg = "periods must be an int or a sequence of ints"
    raise ValueError(msg)


@dataclass(kw_only=True)
class GAMSeasonalityCostFunction(SimpleRootMeanSquareErrorCostFunction):
    """
    Cost function that use the GAM decomposition on both the observations and the model predictions.

    Attributes
    ----------
    observations : Sequence[GAMSeasonalObservation]
        The list of observations to compare against the model predictions.
    n_splines : int
        Number of splines to use for the GAM decomposition, by default 20.
    weights : Sequence[Number]
        Relative weights in the cost function assigned to the trend and seasonal components.
    fit_intercept : bool
        Whether to fit an intercept in the GAM model, by default False.

    """

    seasonal_weights: Sequence[Number]
    observations: Sequence[GAMSeasonalObservation]
    n_splines: int = 20
    fit_intercept: bool = False

    def __post_init__(self: GAMSeasonalityCostFunction) -> None:
        """Check that the kwargs are set."""
        super().__post_init__()
        if not isinstance(self.seasonal_weights, Sequence):
            msg = "Weights must be a sequence of numbers."
            raise TypeError(msg)
        self.seasonal_weights = np.asarray(self.seasonal_weights) / np.sum(self.seasonal_weights)

    def _cost_function(self: GAMSeasonalityCostFunction, args: np.ndarray) -> tuple:
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

        result = []
        for obs in self.observations:
            latitude = obs.observation.cf.indexes[CoordinatesLabels.Y][0]
            longitude = obs.observation.cf.indexes[CoordinatesLabels.X][0]
            layer = obs.observation.cf.indexes[CoordinatesLabels.Z][0]

            prediction = biomass_day if obs.observation_type == DayCycle.DAY else biomass_night
            prediction = prediction.cf.sel(
                {CoordinatesLabels.X: longitude, CoordinatesLabels.Y: latitude, CoordinatesLabels.Z: layer}
            )

            prediction_gam = decompose_gam(
                time=prediction.cf.indexes["T"],
                data=prediction.squeeze().data,
                n_splines=self.n_splines,
                fit_intercept=self.fit_intercept,
            )

            rmse_trend = root_mean_square_error(
                pred=prediction_gam.trend,
                obs=obs.trend,
                root=self.root_mse,
                centered=self.centered_mse,
                normalized=self.normalized_mse,
            )
            rmse_seasonal = root_mean_square_error(
                pred=prediction_gam.seasonal,
                obs=obs.seasonal,
                root=self.root_mse,
                centered=self.centered_mse,
                normalized=self.normalized_mse,
            )

            result.append(rmse_trend * self.seasonal_weights[0] + rmse_seasonal * self.seasonal_weights[1])
        return tuple(result)
