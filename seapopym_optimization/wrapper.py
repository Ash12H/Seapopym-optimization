"""This is the module that wraps the SeapoPym model to automatically create simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.parameter import ForcingParameters, FunctionalGroups, NoTransportParameters
from seapopym.configuration.parameters.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
)
from seapopym.model.no_transport_model import NoTransportModel

if TYPE_CHECKING:
    from collections.abc import Iterable

NO_TRANSPORT_DAY_LAYER_POS = 0
NO_TRANSPORT_NIGHT_LAYER_POS = 1


@dataclass
class FunctionalGroupGeneratorNoTransport:
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
        - tr_max
        - tr_rate
        - inv_lambda_max
        - inv_lambda_rate

    """

    parameters: np.ndarray
    """
    Axes: (functional_group, parameter). The parameters order is : day_layer, night_layer, energy_transfert, tr_max,
    tr_rate, inv_lambda_max, inv_lambda_rate.
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
                "The number of parameters must be 7 : day_layer, night_layer, energy_transfert, tr_max, tr_rate,"
                "inv_lambda_max, inv_lambda_rate.",
            )
            raise ValueError(msg)

        if self.groups_name is None:
            self.groups_name = [f"D{day_layer}N{night_layer}" for day_layer, night_layer in self.parameters[:, 4:6]]
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
        tr_max: float = fg_parameters[3]
        tr_rate: float = fg_parameters[4]
        inv_lambda_max: float = fg_parameters[5]
        inv_lambda_rate: float = fg_parameters[6]

        return FunctionalGroupUnit(
            name=fg_name,
            migratory_type=FunctionalGroupUnitMigratoryParameters(day_layer=day_layer, night_layer=night_layer),
            functional_type=FunctionalGroupUnitRelationParameters(
                inv_lambda_max=inv_lambda_max,
                inv_lambda_rate=inv_lambda_rate,
                temperature_recruitment_rate=tr_rate,
                cohorts_timesteps=[1] * np.ceil(tr_max).astype(int),
                temperature_recruitment_max=tr_max,
            ),
            energy_transfert=energy_transfert,
        )

    def generate(self: FunctionalGroupGeneratorNoTransport) -> FunctionalGroups:
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

        return FunctionalGroups(functional_groups=fgroups)


# TODO(Jules) : Est-ce qu'on peut envelopper cette fonctionnalité ? Comme un wrapper de classe qui retourne un model
# Seapopym.
def model_generator_no_transport(
    forcing_parameters: ForcingParameters, fg_parameters: FunctionalGroupGeneratorNoTransport, **kwargs: dict
) -> NoTransportModel:
    """Generate a NoTransportModel object with the given parameters."""
    return NoTransportModel(
        configuration=NoTransportConfiguration(
            parameters=NoTransportParameters(
                forcing_parameters=forcing_parameters,
                functional_groups_parameters=fg_parameters.generate(),
                **kwargs,
            )
        )
    )
