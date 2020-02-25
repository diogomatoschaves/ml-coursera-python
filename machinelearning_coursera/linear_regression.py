import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, strategy='gradient_descent', max_iter=1000, learning_rate=0.01, normalize=False):

        self.strategy = strategy
        self._theta = None
        self._cost = np.inf
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.threshold = learning_rate * 1E-2
        self.n_iter = 0
        self.final_cost = self._cost
        self.normalize = normalize

    def fit(self, x, y):

        """
        :param x: matrix with m training examples x n features
        :param y: vector with m target values
        :return: None

        Takes the training data and the target values,
        and fits a linear regression model to the data

        """

        if self.normalize:
            x = self.normalize_features(x)

        X = np.c_[np.ones(x.shape[0]), x]
        theta = np.zeros(X.shape[1])

        costs = np.empty(shape=(0, 2))

        i = 0
        try:
            while i < self.max_iter:

                theta = self.gradient_descent(X, y, theta)

                prev_cost = self._cost
                self._cost = self.cost_function(X, y, theta)

                costs = np.vstack((costs, np.array([[i, self._cost]])))

                if abs(self._cost - prev_cost) < self.threshold:
                    break
                if self._cost - prev_cost > 0:
                    self.learning_rate = self.learning_rate * 0.1
                    print('Updated learning rate: {}: {}'.format(i, self.learning_rate))

                i += 1
        except RuntimeWarning:
            print('errored')
            pass

        self._theta = theta
        self.n_iter = i
        self.final_cost = self._cost

        self.plot_costs(costs)

    def predict(self, x):

        if self._theta is None:
            return 'There is no model fitted'

        X = np.c_[np.ones(x.shape[0]), x]

        return self._theta.T @ X.T

    def cost_function(self, X, y, theta):

        """
        :param X: matrix of m training examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :return: cost

        Computes the value of the cost function for a given X, y and theta
        with the following equation:

        J(theta) = (1 / 2m) * (X*theta - y)' (X*theta - y)
        """

        self._cost = (1 / (2 * len(X))) * (X @ theta - y).T @ (X @ theta - y)

        return self._cost

    def gradient_descent(self, X, y, theta):

        """
        :param X: training data matrix of m examples and n + 1 features
        :param y: vector of m target values
        :param theta: vector of n + 1 parameters for target line
        :param learning_rate: learning rate of the model
        :return: Updated theta vector

        Computes the new values of theta according to gradient
        descent formula.

        """

        return theta - (self.learning_rate / X.shape[0]) * X.T @ (X @ theta - y)

    @staticmethod
    def normalize_features(x):
        """
        :param x: matrix of m examples by n features
        :return: normalized features matrix

        """

        return (x - x.mean(axis=0)) / x.std(axis=0)

    @staticmethod
    def plot_costs(costs):
        fig = plt.figure(figsize=(10, 10))

        plt.plot(costs[:, 0], costs[:, 1])

        plt.xlabel('Nr Iterations')
        plt.ylabel('J(\u03B8)')

        plt.show()
