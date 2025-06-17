"""A simple Logbook definition for use with DEAP. We use Pandera to validate the structure of the logbook."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from seapopym_optimization.functional_group.base_functional_group import AbstractFunctionalGroup, FunctionalGroupSet
from seapopym_optimization.functional_group.sobol_initialization import initialize_with_sobol_sampling

if TYPE_CHECKING:
    from collections.abc import Sequence


class LogbookCategory(StrEnum):
    """Enumeration of the logbook categories for the genetic algorithm."""

    PARAMETER = "Parametre"
    FITNESS = "Fitness"
    WEIGHTED_FITNESS = "Weighted_fitness"


class LogbookIndex(StrEnum):
    """Enumeration of the logbook index for the genetic algorithm."""

    GENERATION = "Generation"
    PREVIOUS_GENERATION = "Is_From_Previous_Generation"
    INDIVIDUAL = "Individual"

    def get_index(self: LogbookIndex) -> str:
        """Get the index for the logbook category."""
        return list(LogbookIndex).index(self)


parameter_column_schema = pa.Column(regex=True)
fitness_column_schema = pa.Column(regex=True, nullable=True)
weighted_fitness_column_schema = pa.Column(regex=True, nullable=True)


multiple_index_schema = pa.MultiIndex(
    [
        pa.Index(
            pa.Int,
            name=LogbookIndex.GENERATION,
            nullable=False,
            checks=pa.Check(lambda x: x >= 0, error="Generation index must be non-negative."),
        ),
        pa.Index(
            pa.Bool,
            name=LogbookIndex.PREVIOUS_GENERATION,
            nullable=False,
        ),
        pa.Index(
            pa.Int,
            name=LogbookIndex.INDIVIDUAL,
            nullable=False,
            checks=pa.Check(lambda x: x >= 0, error="Individual index must be non-negative."),
        ),
    ],
    coerce=True,
)


logbook_schema = pa.DataFrameSchema(
    columns={
        (LogbookCategory.PARAMETER, ".*"): parameter_column_schema,
        (LogbookCategory.FITNESS, ".*"): fitness_column_schema,
        (LogbookCategory.WEIGHTED_FITNESS, LogbookCategory.WEIGHTED_FITNESS): weighted_fitness_column_schema,
    },
    index=multiple_index_schema,
    strict=True,
)


class Logbook(DataFrame[logbook_schema]):
    """A simple logbook for tracking generations in a genetic algorithm."""

    @classmethod
    def from_individual(
        cls: type[Logbook],
        generation: int,
        is_from_previous_generation: list[bool],
        individual: list[list],
        parameter_names: list[str],
        fitness_name: list[str],
    ) -> Logbook:
        """
        Create a Logbook from a list of individuals.

        Parameters
        ----------
        generation: int
            The generation number for the individuals.
        is_from_previous_generation: list[bool]
            A list indicating whether each individual is from the previous generation.
        individual: list[list]
            A list of individuals, where each individual is a list of parameter values.
        parameter_names: list[str]
            A list of names for the parameters of the individuals.
        fitness_name: list[str]
            A list of names for the fitness values of the individuals.

        """
        index = pd.MultiIndex.from_arrays(
            [[generation] * len(individual), is_from_previous_generation, range(len(individual))],
            names=[LogbookIndex.GENERATION, LogbookIndex.PREVIOUS_GENERATION, LogbookIndex.INDIVIDUAL],
        )
        columns = pd.MultiIndex.from_tuples(
            [(LogbookCategory.PARAMETER.value, name) for name in parameter_names]
            + [(LogbookCategory.FITNESS.value, name) for name in fitness_name]
            + [(LogbookCategory.WEIGHTED_FITNESS.value, LogbookCategory.WEIGHTED_FITNESS.value)],
            names=["category", "name"],
        )

        data = np.asarray([indiv + list(indiv.fitness.values) + [sum(indiv.fitness.wvalues)] for indiv in individual])

        return cls(data=data, index=index, columns=columns)

    @classmethod
    def from_array(
        cls: type[Logbook],
        generation: Sequence[int],
        is_from_previous_generation: Sequence[bool],
        individual: Sequence[Sequence[float]],
        parameter_names: Sequence[str],
        fitness_name: Sequence[str],
        fitness_values: Sequence[Sequence[float]] | None = None,
        weighted_fitness: Sequence[float] | None = None,
    ) -> Logbook:
        index = pd.MultiIndex.from_arrays(
            [generation, is_from_previous_generation, range(len(individual))],
            names=[LogbookIndex.GENERATION, LogbookIndex.PREVIOUS_GENERATION, LogbookIndex.INDIVIDUAL],
        )
        columns = pd.MultiIndex.from_tuples(
            [(LogbookCategory.PARAMETER, name) for name in parameter_names]
            + [(LogbookCategory.FITNESS, name) for name in fitness_name]
            + [(LogbookCategory.WEIGHTED_FITNESS, LogbookCategory.WEIGHTED_FITNESS)],
            names=["category", "name"],
        )
        fitness_values = fitness_values or np.full((len(individual), len(fitness_name)), np.nan)

        weighted_fitness = weighted_fitness or np.full((len(individual), 1), np.nan)
        data = np.concatenate(
            [
                np.asarray(individual),
                np.asarray(fitness_values).reshape(len(individual), len(fitness_name)),
                np.asarray(weighted_fitness).reshape(len(individual), 1),
            ],
            axis=1,
        )
        return cls(data=data, index=index, columns=columns)

    def append_new_generation(self: Logbook, new_generation: Logbook) -> Logbook:
        """Append a new generation to the logbook."""
        if not isinstance(new_generation, Logbook):
            msg = "new_generation must be a Logbook instance."
            raise TypeError(msg)

        return Logbook(pd.concat([self, new_generation]))


def generate_logbook_with_sobol_sampling(
    functional_group_parameters: Sequence[AbstractFunctionalGroup] | FunctionalGroupSet,
    sample_number: int,
    fitness_name: Sequence[str],
) -> Logbook:
    """
    Generate a Logbook with Sobol sampling.

    Parameters
    ----------
    functional_group_parameters: list[str]
        A list of parameter names for the functional groups to be included in the logbook under the PARAMETER category.
    sample_number: int
        N parameter used by the SALib `sample_sobol` method. The number of generated samples is equal to `N * (D + 2)`,
        where D is the number of parameters.
    fitness_name: list[str]
        A list of fitness names to be included in the logbook under the FITNESS category.

    Returns
    -------
    Logbook
        A Logbook containing the Sobol samples.

    """
    samples = initialize_with_sobol_sampling(functional_group_parameters, sample_number)

    return Logbook.from_array(
        generation=[0] * len(samples),
        is_from_previous_generation=[False] * len(samples),
        individual=samples.to_numpy(),
        parameter_names=samples.columns.tolist(),
        fitness_name=fitness_name,
    )


if __name__ == "__main__":
    # Example usage
    from seapopym_optimization.functional_group.base_functional_group import Parameter
    from seapopym_optimization.functional_group.no_transport_functional_groups import NoTransportFunctionalGroup

    functional_group_set = [
        NoTransportFunctionalGroup(
            name="group1",
            night_layer=0,
            day_layer=1,
            energy_transfert=Parameter("energy_transfert", 0.0, 1.0),
            lambda_temperature_0=0.5,
            gamma_lambda_temperature=0.5,
            tr_0=0.5,
            gamma_tr=0.5,
        )
    ]
    functional_group_set = FunctionalGroupSet(functional_group_set)
    logbook = generate_logbook_with_sobol_sampling(functional_group_set, 5, ["fitness1", "fitness2"])
    print(logbook)
