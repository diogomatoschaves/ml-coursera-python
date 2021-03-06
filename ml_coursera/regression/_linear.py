from ..regression._base_class import Regression
import numpy as np
from ml_coursera.preprocessing import normalize_features


STRATEGY_OPTIONS = {"gradient_descent", "normal_eq"}


class LinearRegression(Regression):

    """
    Class for performing linear regression on a training set.
    """

    def __init__(
        self,
        strategy="gradient_descent",
        max_iter=10000,
        learning_rate=0.01,
        normalize=False,
        reg_param=0,
        reg_method="ridge",
    ):

        Regression.__init__(
            self, max_iter, learning_rate, normalize, reg_param, reg_method
        )
        self.strategy = (
            strategy
            if isinstance(strategy, str) and strategy.lower() in STRATEGY_OPTIONS
            else "gradient_descent"
        )
        self.norm_coeff_ = None

    def fit(self, x, y):

        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values according
        to the chosen strategy

        """

        x, y = super(LinearRegression, self).fit(x, y)

        if self.strategy == "gradient_descent":
            self._gradient_descent_fit(x, y)
        elif self.strategy == "normal_eq":
            self._normal_equation_fit(x, y)
        else:
            print("Invalid strategy")

    def predict(self, x):

        """
        :param x: array of m predicting cases by n features
        :return: predicted values for x based on fitted model

        Predicts the values for a given input array x

        """

        return super(LinearRegression, self).predict(x)

    def score(self, x, y_true):
        """
        :param x: m x n array of m examples by n features
        :param y_true: 1-D array of m examples of true target values
        :return: the score of the model

        Outputs the score of model. Achieved by
        calculating R-squared

        """

        y_true = np.array(y_true).ravel()

        y_pred = self.predict(x).ravel()

        numerator = ((y_true - y_pred) ** 2).sum(axis=0, dtype=np.float64)
        denominator = ((y_true - np.average(y_true, axis=0)) ** 2).sum(
            axis=0, dtype=np.float64
        )

        valid_score = denominator != 0 and numerator != 0

        if valid_score:
            return 1 - numerator / denominator
        else:
            return None

    def _normal_equation_fit(self, x, y):

        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values following
        the normal equation strategy

        """

        X = np.c_[np.ones(x.shape[0]), x]

        theta = np.linalg.inv(X.T @ X) @ X.T @ y

        self.coefficients = theta

    def _cost_function(self, X, y, theta, lambda_=None):

        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :param m: number of training examples
        :return: cost for a given theta

        Computes the value of the cost function for a given X, y and theta
        with the following equation:

        J(theta) = (1 / 2m) * (X*theta - y)' (X*theta - y)

        """

        if not lambda_:
            lambda_ = self.reg_param

        reg_cost = super(LinearRegression, self)._cost_function(X, y, theta, lambda_)

        m = X.shape[0]

        return (1 / (2 * m)) * ((X @ theta - y).T @ (X @ theta - y)) + reg_cost
