"""A simple Logbook definition for use with DEAP. We use Pandera to validate the structure of the logbook."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame


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

    def append_new_generation(self: Logbook, new_generation: Logbook) -> Logbook:
        """Append a new generation to the logbook."""
        if not isinstance(new_generation, Logbook):
            msg = "new_generation must be a Logbook instance."
            raise TypeError(msg)

        return Logbook(pd.concat([self, new_generation]))
