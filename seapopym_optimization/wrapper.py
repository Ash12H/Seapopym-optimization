"""This is the module that wraps the SeapoPym model to automatically create simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from seapopym.configuration.no_transport.parameter import FunctionalGroups
from seapopym.configuration.parameters.parameter_functional_group import (
    FunctionalGroupUnit,
    FunctionalGroupUnitMigratoryParameters,
    FunctionalGroupUnitRelationParameters,
)


@dataclass
class FunctionalGroupGenerator:
    tr_max: float | Iterable = 10.38
    tr_rate: float | Iterable = -0.11
    inv_lambda_max: float | Iterable = 150
    inv_lambda_rate: float | Iterable = 0.15
    day_layer: float | Iterable = 1
    night_layer: float | Iterable = 1
    energy_transfert: float | Iterable = 0.1668
    """
    This class is a wrapper around the SeapoPym model to automatically create functional groups with the given
    parameters.
    """

    def _helper_functional_group_generator(
        self: FunctionalGroupGenerator,
        tr_max: float,
        tr_rate: float,
        inv_lambda_max: float,
        inv_lambda_rate: float,
        day_layer: float,
        night_layer: float,
        energy_transfert: float,
    ) -> FunctionalGroupUnit:
        """Create a single functional group with the given parameters."""
        return FunctionalGroupUnit(
            name=f"D{day_layer}N{night_layer}",
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

    def generate(self: FunctionalGroupGenerator) -> FunctionalGroups:
        """
        Generate a FunctionalGroups object with the given parameters. If the parameters are given as a single value,
        only one functional group will be created with these parameters. If the parameters are given as an iterable,
        the number of values must be the same for all the parameters.
        """
        if not isinstance(self.tr_max, Iterable):
            self.tr_max = [self.tr_max]
        if not isinstance(self.tr_rate, Iterable):
            self.tr_rate = [self.tr_rate]
        if not isinstance(self.inv_lambda_max, Iterable):
            self.inv_lambda_max = [self.inv_lambda_max]
        if not isinstance(self.inv_lambda_rate, Iterable):
            self.inv_lambda_rate = [self.inv_lambda_rate]
        if not isinstance(self.day_layer, Iterable):
            self.day_layer = [self.day_layer]
        if not isinstance(self.night_layer, Iterable):
            self.night_layer = [self.night_layer]
        if not isinstance(self.energy_transfert, Iterable):
            self.energy_transfert = [self.energy_transfert]

        nb_params = [
            len(param)
            for param in [
                self.tr_max,
                self.tr_rate,
                self.inv_lambda_max,
                self.inv_lambda_rate,
                self.day_layer,
                self.night_layer,
                self.energy_transfert,
            ]
        ]

        if len(np.unique(nb_params)) != 1:
            msg = "You must specify a constant number of values for each parameter."
            raise ValueError(msg)

        iterable_params = zip(
            self.tr_max,
            self.tr_rate,
            self.inv_lambda_max,
            self.inv_lambda_rate,
            self.day_layer,
            self.night_layer,
            self.energy_transfert,
        )

        fgroups = [self._helper_functional_group_generator(*params) for params in iterable_params]

        return FunctionalGroups(functional_groups=fgroups)
