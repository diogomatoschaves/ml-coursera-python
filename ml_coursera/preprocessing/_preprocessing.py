import numpy as np
from itertools import product


def normalize_features(x, norm_params_=None):
    """
    :param x: numpy array of m examples by n features
    :param norm_params_: Optional: mean and std to be used in normalization
    :return: numpy array of normalized features

    Normalizes the features matrix

    """

    x = np.array(x)

    [mean, std] = (
        [x.mean(axis=0), x.std(axis=0)] if norm_params_ is None else norm_params_
    )

    with np.errstate(invalid="ignore"):
        normalized_x = (x - mean) / std

    for i, std_val in enumerate(std):
        if std_val == 0:
            normalized_x[:, i] = x[:, i]

    return normalized_x


def feature_mapping(x, order, intercept=False, only_self_terms=False):
    """
    Maps the original features up to the chosen degree.

    Example for initial features a and b and chosen order of 3:

    [a b a^2 ab b^2 a^3 a^2b ab^2 b^3]

    :param x: array like object of m examples by n features
    :param order: order of the polynomial expansion mapping to perform
    :param intercept: If return array should include the intercept column
    :param only_self_terms: if should only include polynomial terms (eg: x, x2, x3, etc)
    :return: array with mapped features
    """
    X = np.array(x).copy()

    n_features = X.shape[1] if len(X.shape) > 1 else 1
    features = [i for i in range(n_features)]

    for i in range(2, order + 1):

        if only_self_terms:

            for j in features:
                # X = np.hstack((X, X[:, j] ** i))
                X = np.c_[X, X[:, j] ** i]

        else:
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
