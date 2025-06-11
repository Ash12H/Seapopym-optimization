"""Sobol sampling initialization for functional group parameters."""

from collections.abc import Sequence

import pandas as pd

from seapopym_optimization.functional_group.base_functional_group import AbstractFunctionalGroup, FunctionalGroupSet


def initialize_with_sobol_sampling(
    functional_group_parameters: Sequence[AbstractFunctionalGroup] | FunctionalGroupSet,
    sample_number: int,
    *,
    calc_second_order: bool = False,
) -> pd.DataFrame:
    """
    Generate Sobol samples for the given functional group parameters.
    This function uses the SALib library to generate samples based on the specified functional group parameters.
    """
    try:
        from SALib import ProblemSpec
    except ImportError as e:
        msg = "SALib is required for Sobol sampling : "
        raise ImportError(msg) from e

    if not isinstance(functional_group_parameters, FunctionalGroupSet):
        functional_group_parameters = FunctionalGroupSet(functional_group_parameters)

    name_and_bounds = functional_group_parameters.unique_functional_groups_parameters_ordered()
    bounds = [[i.lower_bound, i.upper_bound] for i in name_and_bounds.values()]
    sp = ProblemSpec({"names": name_and_bounds.keys(), "bounds": bounds})
    samples = sp.sample_sobol(sample_number, calc_second_order=calc_second_order)
    return pd.DataFrame(samples.samples, columns=name_and_bounds.keys())
