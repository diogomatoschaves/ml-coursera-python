import pytest
import numpy as np
from importlib import import_module
from ml_coursera.preprocessing import normalize_features, feature_mapping


def get_test_fixture(test_case, func):

    module = import_module(
        f".{test_case}", package=f"ml_coursera.tests.fixtures.preprocessing.{func}"
    )
    base_array = getattr(module, "base_array")
    expected_array = getattr(module, "expected_array")

    if func == "mapping":
        order = getattr(module, "order")
        only_self_terms = getattr(module, "only_self_terms")
        return base_array, expected_array, order, only_self_terms

    return base_array, expected_array


fixtures = [
    "single_array_all_ones",
    "single_array_all_zeros",
    "single_array_mixed",
    "multi_array_all_ones",
    "multi_array_mixed",
]


@pytest.mark.parametrize(
    "base_array,expected_array",
    [
        pytest.param(*get_test_fixture(fixture, "normalize"), id=fixture)
        for fixture in fixtures
    ],
)
def test_normalize_features(base_array, expected_array):

    output_array = normalize_features(base_array)

    assert np.allclose(expected_array, output_array, atol=1e-2)


fixtures = [
    "1D_array_all_ones_order_2",
    "1D_array_mixed_order_3",
    "2D_array_all_ones_order_2",
    "2D_array_all_ones_order_3",
    "2D_array_mixed_order_2",
    "2D_array_mixed_order_3",
    "2D_array_mixed_order_5",
    "2D_array_only_self_terms_mixed_5"
]


@pytest.mark.parametrize(
    "base_array,expected_array,order,only_self_terms",
    [
        pytest.param(*get_test_fixture(fixture, "mapping"), id=fixture)
        for fixture in fixtures
    ],
)
def test_feature_mapping(base_array, expected_array, order, only_self_terms):

    output_array = feature_mapping(base_array, order, only_self_terms=only_self_terms)

    print(output_array)
    print(expected_array)

    assert np.allclose(expected_array, output_array, atol=1e-2)
