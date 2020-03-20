from importlib import import_module

import pytest
import numpy as np
import matplotlib.pyplot as plt

from ml_coursera.regression import LogisticRegression
from ml_coursera.tests.mocks.plt import mock_show


@pytest.fixture
def mock_matplotlib(mocker):
    mocker.patch.object(plt, "show", mock_show)


def get_test_fixture(test_case, module, expected_result):

    module = import_module(
        f".{test_case}", package=f"ml_coursera.tests.fixtures.regression.{module}"
    )

    data = getattr(module, "data")
    max_iter = getattr(module, "max_iter")
    normalize = getattr(module, "normalize")
    learning_rate = getattr(module, "learning_rate")
    reg_param = getattr(module, "reg_param")
    expected_result = getattr(module, expected_result)

    return data, max_iter, learning_rate, reg_param, normalize, expected_result


fixtures = [
    "unregularized_2_features",
    "regularized_2_features",
]


class TestLogisticRegression:

    test_input_string = "data,max_iter,learning_rate,reg_param,normalize,expected_result"

    def test_sigmoid_function(self):

        sigmoid = LogisticRegression()._sigmoid

        x = np.array([-np.inf, -1, 0, 1, np.inf])

        assert np.allclose(sigmoid(x), np.array([0, 0.268, 0.5, 0.731, 1]), atol=1e-3)

    @pytest.mark.parametrize(
        "file_name",
        ["unregularized", "regularized"],
    )
    def test_cost_function(self, file_name):

        module = import_module(
            f".cost_function_gradient_{file_name}", package=f"ml_coursera.tests.fixtures.regression.logistic"
        )

        data = getattr(module, "data")
        reg_params = getattr(module, "reg_params")
        test_theta = getattr(module, "test_theta")
        expected_cost = getattr(module, "expected_cost")
        expected_gradient = getattr(module, "expected_gradient")

        m = data.shape[0]

        X = np.c_[np.ones(m), data[:, :-1]]
        y = data[:, -1]

        for reg_param, theta, cost, grad in zip(reg_params, test_theta, expected_cost, expected_gradient):

            reg = LogisticRegression(reg_param=reg_param)

            cost_function = reg._cost_function
            gradient = reg._gradient

            assert np.allclose([cost_function(X, y, theta)], [cost], atol=1e-2)

            assert np.allclose(gradient(X, y, theta)[:5], grad, atol=1e-1)

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "logistic", "expected_theta"), id=fixture
            )
            for fixture in fixtures
        ],
    )
    def test_coefficients(
        self,
        mock_matplotlib,
        data,
        max_iter,
        learning_rate,
        reg_param,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LogisticRegression(
            max_iter=max_iter, learning_rate=learning_rate, normalize=normalize, reg_param=reg_param
        )
        reg.fit(X, y)

        assert np.allclose(reg.coefficients[:6], expected_result, rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "logistic", "expected_predictions"),
                id=fixture,
            )
            for fixture in fixtures
        ],
    )
    def test_predictions(
        self,
        mock_matplotlib,
        data,
        max_iter,
        learning_rate,
        reg_param,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LogisticRegression(
            max_iter=max_iter, learning_rate=learning_rate, normalize=normalize, reg_param=reg_param
        )
        reg.fit(X, y)

        pred = reg.predict(X)

        assert np.allclose(pred.ravel(), expected_result, atol=1e-3)

    @pytest.mark.parametrize(
        test_input_string,
        [
            pytest.param(
                *get_test_fixture(fixture, "logistic", "expected_score"), id=fixture
            )
            for fixture in fixtures
        ],
    )
    def test_score(
        self,
        mock_matplotlib,
        data,
        max_iter,
        learning_rate,
        reg_param,
        normalize,
        expected_result,
    ):

        X = data[:, :-1]
        y = data[:, -1]

        reg = LogisticRegression(
            max_iter=max_iter, learning_rate=learning_rate, normalize=normalize, reg_param=reg_param
        )
        reg.fit(X, y)

        score = reg.score(X, y)

        assert np.allclose([score], [expected_result], atol=1e-3)
