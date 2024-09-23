"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

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
    """Use the Mean Absolute Error (MAE) method or the Mean Squared Error (MSE) method to calculate the cost."""
    total_size = np.flatten(args).size
    func_groups_nb = total_size // nb_parameters
    func_group_matrix = args.reshape(func_groups_nb, nb_parameters)

    fg_parameters = FunctionalGroupGeneratorNoTransport(func_group_matrix, groups_name)

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
