"""This is the module that wraps the SeapoPym model to automatically create simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import xarray as xr
import numpy as np
from seapopym.configuration.abstract_configuration import AbstractFunctionalGroupParameter
from seapopym.configuration.no_transport.kernel_parameter import KernelParameter
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.forcing_parameter import ForcingParameter
 
from seapopym.configuration.no_transport.environment_parameter import EnvironmentParameter
from seapopym.configuration.no_transport.functional_group_parameter import (
    FunctionalGroupUnit,
    MigratoryTypeParameter,
)
from seapopym.configuration import no_transport, acidity
from seapopym.model.no_transport_model import NoTransportModel

from seapopym.configuration.acidity.forcing_parameter import ForcingParameter

from seapopym.configuration.acidity.configuration import AcidityConfiguration
from seapopym.model.acidity_model import AcidityModel

if TYPE_CHECKING:
    from collections.abc import Iterable

NO_TRANSPORT_DAY_LAYER_POS = 0
"""Position of the day layer in the parameters array."""
NO_TRANSPORT_NIGHT_LAYER_POS = 1
"""Position of the night layer in the parameters array."""


@dataclass
class FunctionalGroupGeneratorNoTransport(AbstractFunctionalGroupParameter):
    """
    This class is a wrapper around the SeapoPym model to automatically create functional groups with the given
    parameters. The parameters must be given as a 2D array with the shape (functional_group >=1, parameter == 7).

    Parameters
    ----------
    parameters : np.ndarray
        Axes: (functional_group >=1, parameter == 7). The parameters order is :
        - day_layer
        - night_layer
        - energy_transfert
        - tr_0
        - gamma_tr
        - lambda_temperature_0
        - gamma_lambda_temperature

    """

    parameters: np.ndarray
    """
    Axes: (functional_group, parameter). 
    The parameters order is : day_layer, night_layer, energy_transfert, 
    tr_0, gamma_tr, lambda_temperature_0, gamma_lambda_temperature
    """
    groups_name: list[str] = None

    def __post_init__(self: FunctionalGroupGeneratorNoTransport) -> None:
        """Check the parameters and convert them to a numpy array."""
        if not isinstance(self.parameters, np.ndarray):
            self.parameters = np.array(self.parameters)
        if self.parameters.ndim != 2:
            msg = "The parameters must be a 2D array with the shape (functional_group of shape X, parameter of shape 7)"
            raise ValueError(msg)
        if self.parameters.shape[1] != 7:
            msg = (
                "The number of parameters must be 7 : day_layer, night_layer, energy_transfert, tr_0, gamma_tr, lambda_temperature_0, gamma_lambda_temperature",
            )
            raise ValueError(msg)

        if self.groups_name is None:
            self.groups_name = [f"D{day_layer}N{night_layer}" for day_layer, night_layer in self.parameters[:, :2]]
        elif len(self.groups_name) != self.parameters.shape[0]:
            msg = "The number of names must be the same as the number of functional groups"
            raise ValueError(msg)

    def _helper_functional_group_generator(
        self: FunctionalGroupGeneratorNoTransport,
        fg_parameters: Iterable[int | float],
        fg_name: str,
    ) -> FunctionalGroupUnit:
        """Create a single functional group with the given parameters."""
        day_layer: float = fg_parameters[NO_TRANSPORT_DAY_LAYER_POS]
        night_layer: float = fg_parameters[NO_TRANSPORT_NIGHT_LAYER_POS]
        energy_transfert: float = fg_parameters[2]
        tr_0: float = fg_parameters[3]
        gamma_tr: float = fg_parameters[4]
        lambda_temperature_0: float = fg_parameters[5]
        gamma_lambda_temperature: float = fg_parameters[6]

        return FunctionalGroupUnit(
            name=fg_name,
            migratory_type=MigratoryTypeParameter(day_layer=day_layer, night_layer=night_layer),
            functional_type=no_transport.functional_group_parameter.FunctionalTypeParameter(
                lambda_temperature_0=lambda_temperature_0,
                gamma_lambda_temperature=gamma_lambda_temperature,
                gamma_tr=gamma_tr,
                # cohorts_timesteps=[1] * np.ceil(tr_0).astype(int),
                tr_0=tr_0,
            ),
            energy_transfert=energy_transfert,
        )

    def generate(self: FunctionalGroupGeneratorNoTransport) -> no_transport.functional_group_parameter.FunctionalGroupParameter:
        """
        Generate a FunctionalGroups object with the given parameters. If the parameters are given as a single value,
        only one functional group will be created with these parameters. If the parameters are given as an iterable,
        the number of values must be the same for all the parameters.
        """
        nb_functional_groups = self.parameters.shape[0]
        if nb_functional_groups == 1:
            fgroups = [self._helper_functional_group_generator(self.parameters[0], self.groups_name[0])]
        else:
            fgroups = [
                self._helper_functional_group_generator(self.parameters[i], self.groups_name[i])
                for i in range(nb_functional_groups)
            ]

        return no_transport.functional_group_parameter.FunctionalGroupParameter(functional_groups=fgroups)
    
    def to_dataset(self, timestep: int = 1) -> xr.Dataset:
        """Return all the functional groups as a xarray.Dataset."""
        fg_units = [
            self._helper_functional_group_generator(params, name)
            for params, name in zip(self.parameters, self.groups_name)
        ]
        fg_parameter = no_transport.functional_group_parameter.FunctionalGroupParameter(functional_group=fg_units)
        return fg_parameter.to_dataset(timestep)

@dataclass
class FunctionalGroupGeneratorAcidity(AbstractFunctionalGroupParameter):
    """
    This class is a wrapper around the SeapoPym model to automatically create functional groups with the given
    parameters. The parameters must be given as a 2D array with the shape (functional_group >=1, parameter == 9).

    Parameters
    ----------
    parameters : np.ndarray
        Axes: (functional_group >=1, parameter == 9). The parameters order is :
        - day_layer
        - night_layer
        - energy_transfert
        - tr_0
        - gamma_tr
        - lambda_temperature_0
        - gamma_lambda_temperature
        - lambda_acidity_0
        - gamma_lambda_acidity

    """

    parameters: np.ndarray
    """
    Axes: (functional_group, parameter). The parameters order is : day_layer, night_layer, energy_transfert, tr_0,
    gamma_tr, lambda_temperature_0, gamma_lambda_temperature, lambda_acidity_0, gamma_lambda_acidity
    """
    groups_name: list[str] = None

    def __post_init__(self: FunctionalGroupGeneratorAcidity) -> None:
        """Check the parameters and convert them to a numpy array."""
        if not isinstance(self.parameters, np.ndarray):
            self.parameters = np.array(self.parameters)
        if self.parameters.ndim != 2:
            msg = "The parameters must be a 2D array with the shape (functional_group of shape X, parameter of shape 9)"
            raise ValueError(msg)
        if self.parameters.shape[1] != 9:
            msg = (
                "The number of parameters must be 9 : day_layer, night_layer, energy_transfert, tr_0, gamma_tr,"
                "lambda_temperature_0, gamma_lambda_temperature, lambda_acidity_0, gamma_lambda_acidity",
            )
            raise ValueError(msg)

        if self.groups_name is None:
            self.groups_name = [f"D{day_layer}N{night_layer}" for day_layer, night_layer in self.parameters[:, :2]]
        elif len(self.groups_name) != self.parameters.shape[0]:
            msg = "The number of names must be the same as the number of functional groups"
            raise ValueError(msg)

    def _helper_functional_group_generator(
        self: FunctionalGroupGeneratorAcidity,
        fg_parameters: Iterable[int | float],
        fg_name: str,
    ) -> FunctionalGroupUnit:
        """Create a single functional group with the given parameters."""
        day_layer: float = fg_parameters[NO_TRANSPORT_DAY_LAYER_POS]
        night_layer: float = fg_parameters[NO_TRANSPORT_NIGHT_LAYER_POS]
        energy_transfert: float = fg_parameters[2]
        tr_0: float = fg_parameters[3]
        gamma_tr: float = fg_parameters[4]
        lambda_temperature_0: float = fg_parameters[5]
        gamma_lambda_temperature: float = fg_parameters[6]
        lambda_acidity_0: float = fg_parameters[7]
        gamma_lambda_acidity: float = fg_parameters[8]

        return FunctionalGroupUnit(
            name=fg_name,
            migratory_type=MigratoryTypeParameter(day_layer=day_layer, night_layer=night_layer),
            functional_type=acidity.functional_group_parameter.FunctionalTypeParameter(
                lambda_temperature_0=lambda_temperature_0,
                gamma_lambda_temperature=gamma_lambda_temperature,
                lambda_acidity_0=lambda_acidity_0,
                gamma_lambda_acidity=gamma_lambda_acidity,
                # temperature_recruitment_rate=gamma_tr,
                gamma_tr=gamma_tr,
                # cohorts_timesteps=[1] * np.ceil(tr_0).astype(int),
                # temperature_recruitment_max=tr_0,
                tr_0=tr_0
            ),
            energy_transfert=energy_transfert,
        )
    def to_dataset(self, timestep: int = 1) -> xr.Dataset:
        """Return all the functional groups as a xarray.Dataset."""
        fg_units = [
            self._helper_functional_group_generator(params, name)
            for params, name in zip(self.parameters, self.groups_name)
        ]
        fg_parameter = acidity.functional_group_parameter.FunctionalGroupParameter(functional_group=fg_units)
        return fg_parameter.to_dataset(timestep)


    def generate(self: FunctionalGroupGeneratorAcidity) -> acidity.functional_group_parameter.FunctionalGroupParameter:
        """
        Generate a FunctionalGroups object with the given parameters. If the parameters are given as a single value,
        only one functional group will be created with these parameters. If the parameters are given as an iterable,
        the number of values must be the same for all the parameters.
        """
        nb_functional_groups = self.parameters.shape[0]
        if nb_functional_groups == 1:
            fgroups = [self._helper_functional_group_generator(self.parameters[0], self.groups_name[0])]
        else:
            fgroups = [
                self._helper_functional_group_generator(self.parameters[i], self.groups_name[i])
                for i in range(nb_functional_groups)
            ]

        return acidity.functional_group_parameter.FunctionalGroupParameter(functional_groups=fgroups)

def model_generator_no_transport(
    forcing_parameters: ForcingParameter,
    fg_parameters: FunctionalGroupGeneratorNoTransport,
    environment_parameters: EnvironmentParameter = None,
    kernel_parameters: KernelParameter = None,
) -> NoTransportModel:
    """Generate a NoTransportModel object with the given parameters."""
    if environment_parameters is None:
        environment_parameters = EnvironmentParameter()
    if kernel_parameters is None:
        kernel_parameters = KernelParameter()

    parameters = NoTransportConfiguration(
        forcing=forcing_parameters, 
        functional_group=fg_parameters,
        environment=environment_parameters,
        kernel=kernel_parameters,
        )    
    return NoTransportModel.from_configuration(configuration=parameters)
    # return NoTransportModel(
    #     configuration=NoTransportConfiguration(
    #         parameters=NoTransportParameters(
    #             forcing_parameters=forcing_parameters,
    #             functional_groups_parameters=fg_parameters.generate(),
    #             environment_parameters=environment_parameters,
    #             kernel_parameters=kernel_parameters,
    #         )
    #     )
    # ) (old seapopym version)

def model_generator_acidity(
    forcing_parameters: ForcingParameter,
    fg_parameters: FunctionalGroupGeneratorAcidity,
    environment_parameters: EnvironmentParameter = None,
    kernel_parameters: KernelParameter = None,
) -> AcidityModel:
    """Generate an AcidityModel object with the given parameters."""
    if environment_parameters is None:
        environment_parameters = EnvironmentParameter()
    if kernel_parameters is None:
        kernel_parameters = KernelParameter()

    parameters = AcidityConfiguration(
        forcing=forcing_parameters, 
        functional_group=fg_parameters,
        environment=environment_parameters,
        kernel=kernel_parameters,
        )    
    return AcidityModel.from_configuration(configuration=parameters)
    # return AcidityModel(
    #     configuration=AcidityConfiguration(
    #         parameters=AcidityParameters(
    #             forcing_parameters=forcing_parameters,
    #             functional_groups_parameters=fg_parameters.generate(),
    #             environment_parameters=environment_parameters,
    #             kernel_parameters=kernel_parameters,
    #         )
    #     )
    # )
