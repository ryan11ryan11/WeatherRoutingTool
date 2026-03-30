from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u

from WeatherRoutingTool.algorithms.genetic.problem import RoutingProblem


def make_problem():
    return RoutingProblem(
        departure_time=datetime(2023, 1, 1, 12, 0),
        arrival_time=datetime(2023, 1, 1, 14, 0),
        boat=MagicMock(),
        boat_speed=5.0,
        constraint_list=MagicMock(),
        objectives={"fuel_consumption": 1.0, "arrival_time": 1.0},
    )


def test_get_power_returns_expected_fuel_and_time():
    problem = make_problem()
    route = np.array([
        [0.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [0.0, 2.0, 5.0],
    ])
    route_dict = {
        "courses": np.array([90.0, 90.0]),
        "start_lats": np.array([0.0, 0.0]),
        "start_lons": np.array([0.0, 1.0]),
        "start_times": np.array([
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 14, 0),
        ]),
        "travel_times": np.array([7200.0, 3600.0]) * u.second,
    }

    ship_params = MagicMock()
    ship_params.get_fuel_rate.return_value = np.array([10.0, 10.0]) * u.kg / u.second
    problem.boat.get_ship_parameters.return_value = ship_params

    with patch(
        "WeatherRoutingTool.algorithms.genetic.problem.RouteParams.get_per_waypoint_coords",
        return_value=route_dict,
    ):
        result = problem.get_power(route)

    assert result["fuel_sum"].to_value(u.kg) == pytest.approx(108000.0)
    assert result["time_obj"] == pytest.approx(12960000.0)
    assert result["shipparams"] is ship_params


def test_get_power_clamps_time_difference_to_one_minute():
    problem = make_problem()
    route = np.array([
        [0.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [0.0, 2.0, 5.0],
    ])
    route_dict = {
        "courses": np.array([90.0, 90.0]),
        "start_lats": np.array([0.0, 0.0]),
        "start_lons": np.array([0.0, 1.0]),
        "start_times": np.array([
            datetime(2023, 1, 1, 12, 0),
            datetime(2023, 1, 1, 13, 59, 50),
        ]),
        "travel_times": np.array([7190.0, 10.0]) * u.second,
    }

    ship_params = MagicMock()
    ship_params.get_fuel_rate.return_value = np.array([10.0, 10.0]) * u.kg / u.second
    problem.boat.get_ship_parameters.return_value = ship_params

    with patch(
        "WeatherRoutingTool.algorithms.genetic.problem.RouteParams.get_per_waypoint_coords",
        return_value=route_dict,
    ):
        result = problem.get_power(route)

    assert result["time_obj"] == pytest.approx(1.0)
    assert result["fuel_sum"].to_value(u.kg) == pytest.approx(72000.0)
