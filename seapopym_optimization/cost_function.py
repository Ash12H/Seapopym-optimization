"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import partial

import numpy as np
from seapopym.configuration.no_transport.configuration import NoTransportConfiguration
from seapopym.configuration.no_transport.parameter import ForcingParameters, NoTransportParameters
from seapopym.model.no_transport_model import NoTransportModel

from seapopym_optimization.wrapper import FunctionalGroupGeneratorNoTransport


def cost_function(
    args: np.ndarray,
    nb_parameters: int,
    forcing_parameters: ForcingParameters,
    observations: ...,
    groups_name: list[str] | None = None,
    **kwargs: dict,
) -> tuple[float]:
    """
    Use the Mean Absolute Error (MAE) method or the Mean Squared Error (MSE) method to calculate the cost.

    Parameters
    ----------
    args : np.ndarray
        Parameters to optimize.

    """
    args_flat = np.asarray(args).flatten()
    func_group_matrix = args_flat.reshape(args_flat.size // nb_parameters, nb_parameters)

    fg_parameters = FunctionalGroupGeneratorNoTransport(func_group_matrix, groups_name)

    model = NoTransportModel(
        configuration=NoTransportConfiguration(
            parameters=NoTransportParameters(
                forcing_parameters=forcing_parameters, functional_groups_parameters=fg_parameters.generate(), **kwargs
            )
        )
    )

    model.run()

    predicted_biomass = model.export_biomass()

    # (
    #     energy_transfert,
    #     tr_max,
    #     tr_rate,
    #     inv_lambda_max,
    #     inv_lambda_rate,
    # ) = args
    # fgroups = gen_fgroup(
    #     energy_transfert=energy_transfert,
    #     tr_max=tr_max,
    #     tr_rate=tr_rate,
    #     inv_lambda_max=inv_lambda_max,
    #     inv_lambda_rate=inv_lambda_rate,
    # )

    # setup_model = gen_model(hot_data_parameter, fgroups)
    # setup_model.run()

    # biomass_pred = setup_model.export_biomass().pint.quantify().pint.to("mg / meter ** 2").pint.dequantify()
    # biomass_pred = biomass_pred[0, :, 0, 0].rename("prediction")
    # cost = float(((zoo_obs - biomass_pred) ** 2).mean())

    # return (cost,)

    return predicted_biomass


class GenericCostFunction(ABC):
    """Generic cost function class."""

    def __init__(self, nb_parameters: int, forcing_parameters: ForcingParameters, observations: ...) -> None:
        """
        ...

        Parameters
        ----------
        nb_parameters : int
            Number of parameters by functional group in the simulation.
        forcing_parameters : ForcingParameters
            Forcing parameters.
        observations : ...
            Observations.

        """
        self.nb_parameters = nb_parameters
        self.forcing_parameters = forcing_parameters
        self.observations = observations

    @abstractmethod
    def _cost_function(
        args: np.ndarray, nb_parameters: int, forcing_parameters: ForcingParameters, observations: ...
    ) -> tuple[float]:
        pass

        def generate(self) -> tuple[float]:
            return partial(
                self._cost_function,
                nb_parameters=self.nb_parameters,
                forcing_parameters=self.forcing_parameters,
                observations=self.observations,
            )


# TODO : We should be able to fix some parameters easily
# Utiliser une matrice 2D avec des NONE pour les paramètres à opti et des valeurs pour les paramètres fixé.
# ensuite on rempli la matrice avec les valeurs de args en déroulant la matrice.
