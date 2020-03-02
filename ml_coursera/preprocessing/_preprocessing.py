import numpy as np
from itertools import product


def normalize_features(x):
    """
    :param x: numpy array of m examples by n features
    :return: numpy array of normalized features

    Normalizes the features matrix

    """

    x = np.array(x)

    return (x - x.mean(axis=0)) / x.std(axis=0)


def feature_mapping(x, order, intercept=False):

    """
    :param x: array like object of m examples by n features
    :param order: order of the polynomial expansion mapping to perform
    :param intercept: If return array should include the intercept column
    :return: array with mapped features

    Maps the original features up to the chosen degree.

    Example for initial features a and b and chosen order of 3:

    [a b a^2 ab b^2 a^3 a^2b ab^2 b^3]

    """

    X = x.copy()

    n_features = X.shape[1] if len(X.shape) > 1 else 1
    features = [i for i in range(n_features)]

    for i in range(2, order + 1):

        product_cases = list(product(features, repeat=i))

        product_cases = [tuple(sorted(t)) for t in product_cases]
        product_cases = list(set(product_cases))

        for case in product_cases:
            columns = np.array([x[:, int(col)] for col in case]).T
            columns_prod = np.cumprod(columns, axis=1)[:, -1].reshape(-1, 1)

            X = np.hstack((X, columns_prod))

    if intercept:
        X = np.c_[np.ones(X.shape[0]), X]

    return X
