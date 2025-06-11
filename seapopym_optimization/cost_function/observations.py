"""This module contains several classes to handle observations in the optimization process."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
import xarray as xr
from pandas.tseries.frequencies import to_offset
from seapopym.standard.units import StandardUnitsLabels


class DayCycle(StrEnum):
    """Enum to define the day cycle."""

    DAY = "day"
    NIGHT = "night"


@dataclass
class AbstractObservation(ABC):
    """Abstract class for observations. It is used to define the interface for the observations."""

    name: str
    observation: object


@dataclass
class TimeSeriesObservation(AbstractObservation):
    """The structure used to store the observations as a time series."""

    name: str
    observation: xr.DataArray
    observation_type: DayCycle = DayCycle.DAY
    observation_interval: pd.offsets.BaseOffset | None = None

    def __post_init__(self: TimeSeriesObservation) -> None:
        """Check that the observation data is complient with the format of the predicted biomass."""
        if not isinstance(self.observation, xr.DataArray):
            msg = "Observation must be an xarray DataArray."
            raise TypeError(msg)

        for coord in ["T", "X", "Y", "Z"]:
            if coord not in self.observation.cf.coords:
                msg = f"Coordinate {coord} must be in the observation Dataset."
                raise ValueError(msg)

        try:
            self.observation = self.observation.pint.quantify().pint.to(StandardUnitsLabels.biomass).pint.dequantify()
        except Exception as e:
            msg = (
                f"At least one variable is not convertible to {StandardUnitsLabels.biomass}, which is the unit of the "
                "predicted biomass."
            )
            raise ValueError(msg) from e

        if not isinstance(self.observation_interval, (pd.offsets.BaseOffset, type(None))):
            self.observation_interval = to_offset(self.observation_interval)

        if self.observation_interval is not None:
            self.observation = self.resample_data_by_observation_interval(self.observation)

    def resample_data_by_observation_interval(self: TimeSeriesObservation, data: xr.DataArray) -> xr.DataArray:
        """Resample the data according to the observation type."""
        return data.cf.resample(T=self.observation_interval).mean().cf.dropna("T", how="all")
