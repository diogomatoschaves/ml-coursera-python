import pytest
import numpy as np
import matplotlib.pyplot as plt
from ml_coursera.regression import LinearRegression, LogisticRegression
from ml_coursera.tests.mocks.plt import mock_show


@pytest.fixture
def mock_matplotlib(mocker):
    mock_plt = mocker.patch.object(plt, 'show', mock_show)


def assert_array_equal(first_array, second_array):

    rtol = 1e-5
    atol = 1e-5

    return np.allclose(first_array, second_array, rtol=rtol, atol=atol)


class TestLinearRegression:

    def test_normal_equation(self):

        data = np.array([[1, 2], [2, 3], [3, 4]])

        X = data[:, 0]
        y = data[:, 1]

        reg = LinearRegression(strategy='normal_eq')
        reg.fit(X, y)

        expected_result = np.array([[1.], [1.]])

        assert assert_array_equal(reg.coefficients, expected_result)

    def test_gradient_descent(self, mock_matplotlib):

        data = np.array([[1, 2], [2, 3], [3, 4]])

        X = data[:, 0]
        y = data[:, 1]

        reg = LinearRegression(strategy='gradient_descent')
        reg.fit(X, y)

        expected_result = np.array([[1.], [1.]])

        assert assert_array_equal(reg.coefficients, expected_result)
