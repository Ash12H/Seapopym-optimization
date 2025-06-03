"""This is the module that wraps the SeapoPym model to automatically create simulations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seapopym.configuration.abstract_configuration import (
        AbstractEnvironmentParameter,
        AbstractForcingParameter,
        AbstractKernelParameter,
    )
    from seapopym.model.base_model import BaseModel

    from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet


@dataclass(kw_only=True)
class AbstractModelGenerator(ABC):
    """
    Abstract base class for model generators in SeapoPym optimization.
    This class defines the interface for generating models with specific parameters.
    """

    model_type: type[BaseModel]
    forcing_parameters: AbstractForcingParameter
    environment: AbstractEnvironmentParameter
    kernel: AbstractKernelParameter

    @abstractmethod
    def generate(self: AbstractModelGenerator, functional_group_set: FunctionalGroupSet) -> BaseModel:
        """
        Generate a model of type `self.model_type` with the given parameters.
        This method should be implemented by the subclasses to create the specific model.
        """
