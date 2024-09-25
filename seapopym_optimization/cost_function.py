"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Iterable

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


@dataclass
class GenericCostFunction(ABC):
    """
    Generic cost function class.

    Parameters
    ----------
    nb_parameters : int
        Number of parameters by functional group in the simulation.
    nb_functional_groups : int
        Number of functional groups in the simulation.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : ...
        Observations.
    fixed_parameters : np.ndarray
        Fixed parameters.

    Notes
    -----
    This class is used to create a generic cost function that can be used to optimize the parameters of the SeapoPym
    model. The cost function must be rewritten in the child class following the steps below:
    # - Create a structure (i.e. np.ndarray) to store the parameters of each functional group.

    """

    nb_parameters: int
    nb_functional_groups: int
    forcing_parameters: ForcingParameters
    observations: ...
    fixed_parameters: np.ndarray | None = None

    def __post_init__(self: GenericCostFunction) -> None:
        """Check validity of the class."""
        if self.fixed_parameters is None:
            self.fixed_parameters = np.full((self.nb_functional_groups, self.nb_parameters), np.nan, dtype=float)
        else:
            self.fixed_parameters = np.asarray(self.fixed_parameters, dtype=float)

        if self.fixed_parameters.shape != (self.nb_functional_groups, self.nb_parameters):
            msg = f"Fixed parameters must have the shape ({self.nb_functional_groups}, {self.nb_parameters})"
            raise ValueError(msg)

    def fill_args(self: GenericCostFunction, args: np.ndarray) -> np.ndarray:
        """Fill the fixed parameters in the args. Used to get all the parameters needed for the simulation."""
        args = np.asarray(args, dtype=float)
        args_flat = self.fixed_parameters.flatten()
        args_flat[np.isnan(args_flat)] = args
        return args_flat

    @abstractmethod
    def generate(self: GenericCostFunction) -> Callable[[Iterable[float]], tuple]:
        """
        Generate the partial cost function used for optimization.

        This function must be rewritten in the child class. Example of implementation:

            ```python
            def generate(self: GenericCostFunction) -> Callable[[Iterable[float]], tuple]:
                def cost_function(args: Iterable[float]) -> tuple:  # noqa: ARG001
                    [...some code...]
                    return (cost,)
                return cost_function
        """


# TODO : We should be able to fix some parameters easily
# Utiliser une matrice 2D avec des NONE pour les paramètres à opti et des valeurs pour les paramètres fixé.
# ensuite on rempli la matrice avec les valeurs de args en déroulant la matrice.
