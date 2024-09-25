"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
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


def fill_args(args: np.ndarray, fixed_parameters: np.ndarray) -> np.ndarray:
    """
    Fill the fixed parameters in the args. Used to get all the parameters needed for the simulation.

    Parameters
    ----------
    args : np.ndarray
        Parameters to optimize. This array must be flattened.
    fixed_parameters : np.ndarray
        Fixed parameters. This array must have the shape (nb_functional_groups, nb_parameters).

    Returns
    -------
    np.ndarray
        An array that contains all the parameters needed for the simulation. The shape is the same as the
        fixed_parameters array.

    """
    initial_shape = fixed_parameters.shape
    args = np.asarray(args, dtype=float)
    args_flat = fixed_parameters.flatten()
    args_flat[np.isnan(args_flat)] = args
    return args_flat.reshape(initial_shape)


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
    #TODO(Jules): Add the steps to follow to create a new cost function.

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

    @abstractmethod
    def _cost_function(
        self: GenericCostFunction,
        args: np.ndarray,
        fixed_parameters: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: ...,
    ) -> tuple:
        """
        Calculate the cost of the simulation.

        This function must be rewritten in the child class.
        """

    def generate(self: GenericCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(
            self._cost_function,
            fixed_parameters=self.fixed_parameters,
            forcing_parameters=self.forcing_parameters,
            observations=self.observations,
        )


@dataclass
class NoTransportCostFunction(GenericCostFunction):
    """
    Generator of the cost function for the 'SeapoPym No Transport' model.

    Attributes
    ----------
    nb_parameters : int
        Number of parameters by functional group in the simulation.
    nb_functional_groups : int
        Number of functional groups in the simulation.
    groups_name : list[str]
        List of the functional groups name.
    forcing_parameters : ForcingParameters
        Forcing parameters.
    observations : ...
        Observations.
    fixed_parameters : np.ndarray
        Fixed parameters. The parameters order is : tr_max, tr_rate, inv_lambda_max, inv_lambda_rate, day_layer,
        night_layer, energy_transfert.

    """

    groups_name: list[str] = None

    def __post_init__(self: NoTransportCostFunction) -> None:
        """Check validity of the class."""
        super().__post_init__()

        if self.groups_name is None:
            self.groups_name = [f"FG_{i}" for i in range(self.nb_functional_groups)]

    def _cost_function(
        self: NoTransportCostFunction,
        args: np.ndarray,
        fixed_parameters: np.ndarray,
        forcing_parameters: ForcingParameters,
        observations: ...,
        groups_name: list[str],
        **kwargs: dict,
    ) -> tuple:
        args = fill_args(args, fixed_parameters)
        fg_parameters = FunctionalGroupGeneratorNoTransport(args, groups_name)

        model = NoTransportModel(
            configuration=NoTransportConfiguration(
                parameters=NoTransportParameters(
                    forcing_parameters=forcing_parameters,
                    functional_groups_parameters=fg_parameters.generate(),
                    **kwargs,
                )
            )
        )

        model.run()

        predicted_biomass = model.export_biomass()
        return (predicted_biomass,)

    def generate(self: NoTransportCostFunction) -> Callable[[Iterable[float]], tuple]:
        """Generate the partial cost function used for optimization."""
        return partial(
            self._cost_function,
            fixed_parameters=self.fixed_parameters,
            forcing_parameters=self.forcing_parameters,
            observations=self.observations,
            groups_name=self.groups_name,
        )
