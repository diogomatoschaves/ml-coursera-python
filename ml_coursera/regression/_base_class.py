import matplotlib.pyplot as plt
import numpy as np
from abc import ABCMeta, abstractmethod
from ml_coursera.preprocessing import normalize_features
from ml_coursera.utils import plot_costs

REGULARIZATION_OPTIONS = {"ridge", "lasso"}


class Regression(metaclass=ABCMeta):
    """
    Base class for linear and logistic regression
    """

    def __init__(self, max_iter, learning_rate, normalize, reg_param, reg_method):

        self.coefficients = None
        self._cost = np.inf
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = learning_rate * 1e-4
        self.n_iter = None
        self.final_cost = None
        self.normalize = normalize
        self.norm_params_ = None
        self.reg_param = (
            reg_param if isinstance(reg_param, (int, float)) and reg_param >= 0 else 0
        )
        self.reg_method = (
            reg_method
            if isinstance(reg_method, str)
            and reg_method.lower() in REGULARIZATION_OPTIONS
            else "ridge"
        )

    def fit(self, x, y):

        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values according
        to the chosen strategy

        """

        x = np.array(x)
        y = np.array(y)

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if self.normalize:
            self.norm_params_ = np.array([x.mean(axis=0), x.std(axis=0)])
            x = normalize_features(x)

        return x, y

    def predict(self, x):

        """
        :param x: array of m predicting cases by n features
        :return: predicted values for x based on fitted model

        Predicts the values for a given input array x

        """

        if self.coefficients is None:
            print("There is no model fitted")
            return None

        if self.normalize:
            x = normalize_features(x, self.norm_params_)

        X = np.c_[np.ones(x.shape[0]), x]

        return X @ self.coefficients

    @abstractmethod
    def score(self, x, y_true):
        """
        :param x: m x n array of m examples by n features
        :param y_true: 1-D array of m examples of true target values
        :return: the score of the model

        Outputs the score of model.

        """

        raise NotImplementedError

    def _gradient_descent_fit(self, x, y):
        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values following
        the gradient descent strategy

        """

        X = np.c_[np.ones(x.shape[0]), x]
        theta = np.zeros(X.shape[1]).reshape(-1, 1)

        costs = np.empty(shape=(0, 2))

        m = X.shape[0]

        i = 0
        while i < self.max_iter:

            prev_cost = self._cost
            self._cost = self._regularized_cost(
                self._cost_function(X, y, theta, m), theta, m
            )

            reg_theta = self._regularized_theta(theta, m)

            grad = self.learning_rate * self._gradient(X, y, theta, m)

            theta = reg_theta - grad

            costs = np.vstack((costs, np.array([[i, self._cost]])))

            if self._cost - prev_cost > 0 and i > 20:
                break

            elif self._cost - prev_cost > 0:
                self.learning_rate = self.learning_rate * 0.1
                self.threshold = self.learning_rate * 1e-4
                print("Updated learning rate: {}: {}".format(i, self.learning_rate))

            i += 1

        self.coefficients = theta
        self.n_iter = i
        self.final_cost = self._cost

        plot_costs(costs)

    def _gradient(self, X, y, theta, m):

        """
        :param X: training data matrix of m examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters of target model
        :param m: number of training examples
        :return: gradient for a given theta

        Computes the new values of theta according to gradient
        descent formula.

        """

        return (1 / m) * X.T @ (self._hypothesis(X @ theta) - y)

    def _regularized_theta(self, theta, m):
        """
        :param theta: vector of n + 1 parameters of target model
        :param m: number of training examples
        :return: theta vector with regularization

        Subtracts the regularization term from theta for the Ridge
        Regularization method, or leaves it unchanged for the
        Lasso method.

        """

        if self.reg_method == "ridge":
            reg_term = np.ones(shape=theta.shape) * (
                self.learning_rate * self.reg_param / m
            )
            reg_term[0] = 0
            return theta * (1 - reg_term)
        else:
            return theta

    def _regularized_cost(self, cost, theta, m):
        """
        :param cost: Unregularized cost for a given X and theta
        :param theta: Current hypothesis theta
        :param m: number of training examples
        :return: Updated cost with regularized term

        Returns the cost with the regularization term according to the
        selected option (Ridge or Lasso)

        """

        if self.reg_method == "ridge":
            return cost + (self.reg_param / (2 * m)) * theta[1:].T @ theta[1:]
        elif self.reg_method == "lasso":
            return cost + (self.reg_param / (2 * m)) * theta[1:].sum()
        else:
            return cost

    @abstractmethod
    def _cost_function(self, X, y, theta, m):
        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :param m: number of training examples
        :return: cost

        Computes the value of the cost function for a given X, y and theta

        To be implemented on each child class.

        """

        raise NotImplementedError

    def _hypothesis(self, matrix):
        """
        :param matrix: receives as input the result of X . theta
        :return: same object as received

        This implementation is for the base case, of the hypothesis being
        equivalent to X . theta. For Logistic regression, the hypothesis
        is the return value of the sigmoid function g(X . theta), which needs
        to be implemented on that class.

        """

        return matrix
