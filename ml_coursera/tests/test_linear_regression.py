from importlib import import_module

import pytest
import numpy as np
import matplotlib.pyplot as plt
from ml_coursera.regression import (
    LinearRegression,
    STRATEGY_OPTIONS,
)
from ml_coursera.tests.mocks.plt import mock_show


@pytest.fixture
def mock_matplotlib(mocker):
    mocker.patch.object(plt, "show", mock_show)


def get_test_fixture(test_case, module, expected_result):

    module = import_module(
        f".{test_case}", package=f"ml_coursera.tests.fixtures.regression.{module}"
    )

    data = getattr(module, "data")
    strategy = getattr(module, "strategy")
    max_iter = getattr(module, "max_iter")
    normalize = getattr(module, "normalize")
    learning_rate = getattr(module, "learning_rate")
    expected_result = getattr(module, expected_result)

    return data, strategy, max_iter, learning_rate, normalize, expected_result


fixtures = [
    "single_array_gradient_descent",
    "single_array_normal_equation",
    "single_array_wrong_input_strategy",
    "multi_array_gradient_descent",
    "multi_array_normal_equation",
]


class TestLinearRegression:

    test_input_string = "data,strategy,max_iter,learning_rate,normalize,expected_result"

    def test_cost_function(self):

        from .fixtures.regression.linear.cost_function_gradient import (
            data,
            theta_1,
            theta_2,
            expected_cost_1,
            expected_cost_2,
        )

        cost_function = LinearRegression()._cost_function

        m = data.shape[0]

        X = np.c_[np.ones(m), data[:, 0:-1]]
        y = data[:, -1]

        assert np.allclose(
            [cost_function(X, y, theta_1)], [expected_cost_1], atol=1e-2
        )
        assert np.allclose(
            [cost_function(X, y, theta_2)], [expected_cost_2], atol=1e-2
        )

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "linear", "expected_theta"), id=fixture
            )
            for fixture in fixtures
        ],
    )
    def test_coefficients(
        self,
        mock_matplotlib,
        data,
        strategy,
        max_iter,
        learning_rate,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LinearRegression(
            strategy=strategy,
            max_iter=max_iter,
            learning_rate=learning_rate,
            normalize=normalize,
        )
        reg.fit(X, y)

        assert np.allclose(reg.coefficients, expected_result, rtol=1e-3, atol=1e-3)

        assert (
            reg.strategy == strategy
            if strategy in STRATEGY_OPTIONS
            else reg.strategy == "gradient_descent"
        )

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "linear", "expected_predictions"), id=fixture
            )
            for fixture in fixtures
        ],
    )
    def test_predictions(
        self,
        mock_matplotlib,
        data,
        strategy,
        max_iter,
        learning_rate,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LinearRegression(
            strategy=strategy,
            max_iter=max_iter,
            learning_rate=learning_rate,
            normalize=normalize,
        )
        reg.fit(X, y)

        pred = reg.predict(X)

        assert np.allclose(pred.ravel(), expected_result, atol=1e-3)

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "linear", "expected_score"), id=fixture
            )
            for fixture in fixtures
        ],
    )
    def test_score(
        self,
        mock_matplotlib,
        data,
        strategy,
        max_iter,
        learning_rate,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LinearRegression(
            strategy=strategy,
            max_iter=max_iter,
            learning_rate=learning_rate,
            normalize=normalize,
        )
        reg.fit(X, y)

        score = reg.score(X, y)

        assert np.allclose([score], [expected_result], atol=1e-3)
