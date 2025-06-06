"""Base class for viewers in the optimization process."""

from abc import ABC
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from seapopym_optimization.cost_function.observations import AbstractObservation
from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet
from seapopym_optimization.model_generator.base_model_generator import AbstractModelGenerator


@dataclass
class AbstractViewer(ABC):
    """Base class for parameters of a genetic algorithm."""

    logbook: pd.DataFrame
    functional_group_set: FunctionalGroupSet
    model_generator: AbstractModelGenerator
    observations: Sequence[AbstractObservation]
