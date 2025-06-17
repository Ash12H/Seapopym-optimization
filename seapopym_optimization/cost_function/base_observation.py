"""Base class for observations in the seapopym optimization module."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import StrEnum


class DayCycle(StrEnum):
    """Enum to define the day cycle."""

    DAY = "day"
    NIGHT = "night"


@dataclass(kw_only=True)
class AbstractObservation(ABC):
    """Abstract class for observations. It is used to define the interface for the observations."""

    name: str
    observation: object
