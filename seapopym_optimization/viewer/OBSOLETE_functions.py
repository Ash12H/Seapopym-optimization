# NOTE(Jules) : Unused functions are not removed, as they might be useful in the future.


def correlation_coefficient(
    self: Observation,
    predicted: xr.Dataset,
    day_layer: Sequence[int],
    night_layer: Sequence[int],
    *,
    corr_dim: str = "time",
) -> tuple[float | None, float | None]:
    """Return the correlation coefficient of the predicted and observed biomass."""
    aggregated_prediction = self._helper_day_night_apply(predicted, day_layer, night_layer)
    correlation_day = None
    correlation_night = None
    if "day" in self.observation:
        correlation_day = xr.corr(aggregated_prediction["day"], self.observation["day"], dim=corr_dim)
    if "night" in self.observation:
        correlation_night = xr.corr(aggregated_prediction["night"], self.observation["night"], dim=corr_dim)
    return correlation_day, correlation_night


def normalized_standard_deviation(
    self: Observation, predicted: xr.Dataset, day_layer: Sequence[int], night_layer: Sequence[int]
) -> tuple[float | None, float | None]:
    """Return the normalized standard deviation of the predicted and observed biomass."""
    aggregated_prediction = self._helper_day_night_apply(predicted, day_layer, night_layer)
    normalized_standard_deviation_day = None
    normalized_standard_deviation_night = None
    if "day" in self.observation:
        normalized_standard_deviation_day = aggregated_prediction["day"].std() / self.observation["day"].std()
    if "night" in self.observation:
        normalized_standard_deviation_night = aggregated_prediction["night"].std() / self.observation["night"].std()
    return normalized_standard_deviation_day, normalized_standard_deviation_night


# TODO(Jules): Add bias
def bias(self: Observation, predicted: xr.Dataset, day_layer: Sequence[int], night_layer: Sequence[int]) -> None:
    """Return the bias of the predicted and observed biomass."""
    raise NotImplementedError("The bias is not implemented yet.")
