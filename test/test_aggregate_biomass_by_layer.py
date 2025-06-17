import xarray as xr
from seapopym.standard.labels import CoordinatesLabels

from seapopym_optimization.cost_function.simple_rmse_cost_function import aggregate_biomass_by_layer


def test_aggregate_biomass_by_layer():
    biomass = xr.DataArray(
        data=[10, 20, 30],
        dims=[CoordinatesLabels.functional_group],
        coords={CoordinatesLabels.functional_group: ["fg1", "fg2", "fg3"]},
        name="biomass",
    )
    position = [0, 1, 0]
    layer_coordinates = [0, 1, 2]

    agg = aggregate_biomass_by_layer(
        data=biomass,
        position=position,
        name="agg_biomass",
        layer_coordinates=layer_coordinates,
        layer_coordinates_name="layer",
    )

    expected = xr.DataArray(
        data=[40, 20, 0],  # layer 0: fg1+fg3=10+30=40, layer 1: fg2=20
        dims=["layer"],
        coords={"layer": [0, 1, 2]},
        name="agg_biomass",
    )

    assert agg.equals(expected), f"Results :\n{agg}\nWanted :\n{expected}"


def test_aggregate_biomass_by_layer_with_nan():
    biomass = xr.DataArray(
        data=[10, 20, None, None],
        dims=[CoordinatesLabels.functional_group],
        coords={CoordinatesLabels.functional_group: ["fg1", "fg2", "fg3", "fg4"]},
        name="biomass",
    )
    position = [0, 1, 0, 2]
    layer_coordinates = [0, 1, 2]

    agg = aggregate_biomass_by_layer(
        data=biomass,
        position=position,
        name="agg_biomass",
        layer_coordinates=layer_coordinates,
        layer_coordinates_name="layer",
    )

    expected = xr.DataArray(
        data=[10, 20, 0],
        dims=["layer"],
        coords={"layer": [0, 1, 2]},
        name="agg_biomass",
    )

    assert agg.equals(expected), f"Results :\n{agg}\nWanted :\n{expected}"
