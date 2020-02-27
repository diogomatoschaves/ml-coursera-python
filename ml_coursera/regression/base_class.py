import matplotlib.pyplot as plt
import numpy as np


class Regression:

    """
    Base class for linear and logistic regression
    """

    def __init__(self, max_iter, learning_rate, normalize):

        self.coefficients = None
        self._cost = np.inf
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = learning_rate * 1e-3
        self.n_iter = None
        self.final_cost = None
        self.normalize = normalize

    def fit(self, x, y):

        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Fits the training data and the target values according
        to the chosen strategy

        """

        self._gradient_descent_fit(x, y)

    def predict(self, x):

        """
        :param x: array of m predicting cases by n features
        :return: predicted values for x based on fitted model

        Predicts the values for a given input array x

        """

        raise NotImplementedError

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

        if self.normalize:
            x = self.normalize_features(x)

        X = np.c_[np.ones(x.shape[0]), x]
        theta = np.zeros(X.shape[1])

        costs = np.empty(shape=(0, 2))

        i = 0
        while i < self.max_iter:

            prev_cost = self._cost
            self._cost = self._cost_function(X, y, theta)
            theta = theta - self.learning_rate * self._gradient(X, y, theta)

            costs = np.vstack((costs, np.array([[i, self._cost]])))

            if abs(self._cost - prev_cost) < self.threshold:
                break

            if self._cost - prev_cost > 0:
                self.learning_rate = self.learning_rate * 0.1
                self.threshold = self.learning_rate * 1e-3
                print("Updated learning rate: {}: {}".format(i, self.learning_rate))

            i += 1

        self.coefficients = theta
        self.n_iter = i
        self.final_cost = self._cost

        self.plot_costs(costs)

    def _gradient(self, X, y, theta):

        """
        :param X: training data matrix of m examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :return: gradient for a given theta

        Computes the new values of theta according to gradient
        descent formula.

        """

        return (1 / X.shape[0]) * X.T @ (self._hypothesis(X @ theta) - y)

    def _cost_function(self, X, y, theta):
        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
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

    @staticmethod
    def normalize_features(x):
        """
        :param x: numpy array of m examples by n features
        :return: numpy array of normalized features

        Normalizes the features matrix

        """

        return (x - x.mean(axis=0)) / x.std(axis=0)

    @staticmethod
    def plot_costs(costs):
        """
        :param costs: 2-D array, consisting of the
        number of iterations and associated cost
        :return: None
        """

        plt.figure(figsize=(10, 10))

        plt.plot(costs[:, 0], costs[:, 1])

        plt.xlabel("Nr Iterations")
        plt.ylabel("J(\u03B8)")

        plt.show()
