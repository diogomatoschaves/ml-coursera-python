from ..regression._base_class import Regression
import numpy as np
from ml_coursera.preprocessing import normalize_features


class LogisticRegression(Regression):

    """
    Class for performing logistic regression on a training set.
    """

    def __init__(
        self,
        max_iter=10000,
        learning_rate=0.01,
        normalize=False,
        reg_param=0,
        reg_method="ridge",
    ):

        Regression.__init__(
            self, max_iter, learning_rate, normalize, reg_param, reg_method
        )

    def fit(self, x, y):

        x, y = super(LogisticRegression, self).fit(x, y)

        self._gradient_descent_fit(x, y)

    def predict(self, x):

        """
        :param x: array of m predicting cases by n features
        :return: predicted values for x based on fitted model

        Predicts the values for a given input array x

        """

        predictions = super(LogisticRegression, self).predict(x)

        return np.array(list(map(lambda pred: 1 if pred > 0 else 0, predictions)))

    def score(self, x, y_true):

        """
        :param x: m x n array of m examples by n features
        :param y_true: 1-D array of m examples of true target values
        :return: the score of the model

        Outputs the score of model. Achieved by averaging the
        accuracy of each training example

        """

        y_pred = self.predict(x)

        score = (y_true == y_pred).mean()

        return score

    def _cost_function(self, X, y, theta, lambda_=None):

        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :param m: number of training examples
        :return: cost

        Computes the value of the cost function for a given X, y and theta
        with the following equation:

        J(theta) = (1 / m) * (- y' . log(h) - (1 - y)' . log(1 - h))

        """

        if not lambda_:
            lambda_ = self.reg_param

        reg_cost = super(LogisticRegression, self)._cost_function(X, y, theta, lambda_)

        m = X.shape[0]

        return (1 / m) * (
            -1 * y.T @ np.log(self._sigmoid(X @ theta))
            - (1 - y).T @ np.log(1 - self._sigmoid(X @ theta))
        ) + reg_cost

    def _hypothesis(self, matrix):

        """
        :param matrix: receives as input the result of X . theta
        :return: calculates the sigmoid element wise on the input array

        The hypothesis function is the value of the sigmoid function
        for X . theta g(X . theta).

        """

        return self._sigmoid(matrix)

    @staticmethod
    def _sigmoid(z):
        """
        :param z: value or array like object
        :return: an array or scalar of the applied result

        Calculates the sigmoid values.

        """

        return 1 / (1 + np.exp(-z))
