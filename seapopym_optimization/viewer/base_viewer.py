"""Base class for viewers in the optimization process."""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from seapopym_optimization.cost_function.base_observation import AbstractObservation
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.model_generator.base_model_generator import AbstractModelGenerator


@dataclass
class AbstractViewer(ABC):
    """
    Base class for parameters of a genetic algorithm.

    Attributes
    ----------
    logbook : pd.DataFrame
        DataFrame containing the log of the optimization process.
    functional_group_set : FunctionalGroupSet
        Set of functional groups used in the optimization.
    model_generator : AbstractModelGenerator
        Model generator used to create models for the optimization.
    observations : Sequence[AbstractObservation]
        Sequence of observations used in the optimization process.

    """

    logbook: pd.DataFrame
    functional_group_set: FunctionalGroupSet
    model_generator: AbstractModelGenerator
    observations: Sequence[AbstractObservation]
