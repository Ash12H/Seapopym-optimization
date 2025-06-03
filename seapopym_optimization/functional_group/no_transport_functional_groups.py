"""This module contains the cost function used to optimize the parameters of the SeapoPym model."""

from __future__ import annotations

from dataclasses import dataclass

from seapopym_optimization.functional_group.base_functional_group import AbstractFunctionalGroup, Parameter


@dataclass
class FunctionalGroupOptimizeNoTransport(AbstractFunctionalGroup):
    """The parameters of a functional group as they are defined in the SeapoPym NoTransport model."""

    day_layer: float | Parameter
    night_layer: float | Parameter
    energy_coefficient: float | Parameter
    tr_max: float | Parameter
    tr_rate: float | Parameter
    inv_lambda_max: float | Parameter
    inv_lambda_rate: float | Parameter


if __name__ == "__main__":
    from seapopym_optimization.functional_group.base_functional_group import FunctionalGroupSet

    # Example usage
    fg1 = FunctionalGroupOptimizeNoTransport(
        name="FG1",
        day_layer=Parameter(name="day_layer", lower_bound=0, upper_bound=10),
        night_layer=1,
        energy_coefficient=Parameter(name="FG1_energy_coefficient", lower_bound=0, upper_bound=1),
        tr_max=Parameter(name="tr_max", lower_bound=0, upper_bound=100),
        tr_rate=Parameter(name="tr_rate", lower_bound=0, upper_bound=1),
        inv_lambda_max=Parameter(name="inv_lambda_max", lower_bound=0, upper_bound=10),
        inv_lambda_rate=Parameter(name="inv_lambda_rate", lower_bound=0, upper_bound=1),
    )

    fg2 = FunctionalGroupOptimizeNoTransport(
        name="FG2",
        day_layer=Parameter(name="day_layer", lower_bound=0, upper_bound=10),
        night_layer=2,
        energy_coefficient=Parameter(name="FG2_energy_coefficient", lower_bound=0, upper_bound=1),
        tr_max=Parameter(name="tr_max", lower_bound=0, upper_bound=100),
        tr_rate=Parameter(name="tr_rate", lower_bound=0, upper_bound=1),
        inv_lambda_max=Parameter(name="inv_lambda_max", lower_bound=0, upper_bound=10),
        inv_lambda_rate=Parameter(name="inv_lambda_rate", lower_bound=0, upper_bound=1),
    )

    all_groups = FunctionalGroupSet(functional_groups=[fg1, fg2])
    print(all_groups.generate([5, 5, 50, 0.1, 5, 0.5, 66]))
