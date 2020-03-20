import numpy as np
from abc import ABCMeta, abstractmethod
from ml_coursera.preprocessing import normalize_features
from ml_coursera.utils import plot_costs, plot_learning_curve

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

    def learning_curve(self, x, y, x_test, y_test):
        """
        Calculates and plots the learning curve for a given training and test sets
        by splitting the training set in 20 iterations.

        :param x: training features
        :param y: training labels
        :param x_test: test features
        :param y_test: test labels
        :return: training and test errors
        """

        m = x.shape[0]

        interval = np.floor(m / 20) if m > 20 else 1

        m_lst = np.arange(1, m + 1)[(np.arange(0, m) % interval) == 0]

        err_train = np.zeros(m_lst.size)
        err_test = np.zeros(m_lst.size)

        for i, n in enumerate(m_lst):

            x_subset = x[:n, :]
            y_subset = y[:n]

            theta = self._gradient_descent_fit(x_subset, y_subset, plot_costs_iter=False)

            x_subset_adj = np.c_[np.ones(n), x_subset]
            x_test_adj = np.c_[np.ones(x_test.shape[0]), x_test]

            err_train[i] = self._cost_function(x_subset_adj, y_subset, theta, lambda_=0)
            err_test[i] = self._cost_function(x_test_adj, y_test, theta, lambda_=0)

        plot_learning_curve(err_train, err_test, m_lst)

        return err_train, err_test

    def optimize_reg_param(self, x, y, x_test, y_test):

        lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300])

        err_train = np.zeros(len(lambda_vec))
        err_test = np.zeros(len(lambda_vec))

        for i, l in enumerate(lambda_vec):

            theta = self._gradient_descent_fit(x, y, plot_costs_iter=False, lambda_=l)

            x_adj = np.c_[np.ones(x.shape[0]), x]
            x_test_adj = np.c_[np.ones(x_test.shape[0]), x_test]

            err_train[i] = self._cost_function(x_adj, y, theta, lambda_=0)
            err_test[i] = self._cost_function(x_test_adj, y_test, theta, lambda_=0)

        optimal_lambda = lambda_vec[np.argmin(err_test)]

        plot_learning_curve(err_train, err_test, lambda_vec)

        return optimal_lambda

    def _gradient_descent_fit(self, x, y, plot_costs_iter=True, learning_rate=None, lambda_=None):
        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values following
        the gradient descent strategy

        """

        if not learning_rate:
            learning_rate = self.learning_rate

        if not lambda_:
            lambda_ = self.reg_param

        X = np.c_[np.ones(x.shape[0]), x]
        theta = np.zeros(X.shape[1]).reshape(-1, 1)

        costs = np.empty(shape=(0, 2))

        m = X.shape[0]

        i = 0
        while i < self.max_iter:

            prev_cost = self._cost
            self._cost = self._cost_function(X, y, theta, lambda_)

            theta = theta - learning_rate * self._gradient(X, y, theta, lambda_)

            costs = np.vstack((costs, np.array([[i, self._cost]])))

            if self._cost - prev_cost > 0 and i > 20:
                break

            elif self._cost - prev_cost > 0:
                learning_rate = learning_rate * 0.1
                self.threshold = learning_rate * 1e-4

            i += 1

        self.coefficients = theta
        self.n_iter = i
        self.final_cost = self._cost

        if plot_costs_iter:
            plot_costs(costs)

        return theta

    def _gradient(self, X, y, theta, lambda_=None):

        """
        :param X: training data matrix of m examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters of target model
        :param m: number of training examples
        :return: gradient for a given theta

        Computes the new values of theta according to gradient
        descent formula.

        """

        if not lambda_:
            lambda_ = self.reg_param

        m = X.shape[0]

        if self.reg_method == "ridge":
            reg_term = np.ones(shape=theta.shape) * (lambda_ / m) * theta
            reg_term[0] = 0
        else:
            reg_term = np.zeros(shape=theta.shape)

        return (1 / m) * X.T @ (self._hypothesis(X @ theta) - y) + reg_term

    def _cost_function(self, X, y, theta, lambda_=None):
        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :param m: number of training examples
        :return: cost

        Computes the value of the cost function for a given X, y and theta

        To be implemented on each child class.

        """

        if not lambda_:
            lambda_ = self.reg_param

        m = X.shape[0]

        if self.reg_method == "ridge":
            return (lambda_ / (2 * m)) * theta[1:].T @ theta[1:]
        elif self.reg_method == "lasso":
            return (lambda_ / (2 * m)) * theta[1:].sum()
        else:
            return 0

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
