"""This module provides a function to generate random float values within a specified range."""

from random import uniform

MAXIMUM_INIT_TRY = 1000


def random_uniform_exclusive(lower: float, upper: float) -> float:
    """
    Generate a random float value between `lower` and `upper` bounds, excluding the bounds themselves.
    If the random value equals either bound, it will retry until a valid value is found or the maximum number of tries
    is reached.

    Parameters
    ----------
    lower: float
        The lower bound of the range.
    upper: float
        The upper bound of the range.

    Returns
    -------
    float
        A random float value between `lower` and `upper`, excluding the bounds.

    Raises
    ------
    ValueError
        If the maximum number of tries is reached without finding a valid value.

    """
    count = 0
    while count < MAXIMUM_INIT_TRY:
        value = uniform(lower, upper)  # noqa: S311
        if value not in (lower, upper):
            return value
        count += 1
    msg = "Random parameter initialization reach maximum try."
    raise ValueError(msg)
