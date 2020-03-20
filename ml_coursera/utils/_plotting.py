import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing import feature_mapping


def plot_data(x, y, possible_labels=(0, 1), feature_names=None, logistic=True):

    """
    :param x: matrix of data points of size m training examples by 2 features
    :param y: vector of labels of shape (m, ).
    :param possible_labels: possible categories in y vector. Defaults to 0 and 1
    :param feature_names: names of features. Optional
    :return: None

    Plots the data points according to the labels

    """

    x = np.array(x)

    y = np.array(y).reshape(-1, 1)

    markers = ["o", "x"]

    plt.figure(figsize=(8, 6))

    if not logistic or x.shape[1] == 1:
        plt.scatter(x[:, 0], y[:, 0], marker=markers[0])
    else:
        labels = [(y == lbl) for lbl in possible_labels]
        for i, label in enumerate(labels):
            plt.scatter(x[label[:, 0], 0], x[label[:, 0], 1], marker=markers[i])

    if isinstance(feature_names, list) and len(feature_names) >= 2:
        plt.xlabel(feature_names[0])
        plt.ylabel(feature_names[1])
    else:
        plt.xlabel("feature 1")
        plt.ylabel("feature 2")


def plot_decision_boundary(
    theta,
    x,
    y,
    possible_labels=(0, 1),
    feature_names=None,
    legend=("positive", "negative"),
):
    """
    :param theta: parameters for logistic regression. A vector of shape (n+1, ).
    :param x: The input dataset. X is assumed to be a either:
            1) Mx2 matrix
            2) MxN, N > 2 matrix, where the N > 2 columns are the
                higher order terms of the first 2 features.
    :param y: vector of data labels of shape (m, ).
    :param possible_labels: possible categories in y vector. Defaults to 0 and 1
    :param feature_names: names of features, optional
    :param legend: labels for legend

    Plots the data points X and y into a new figure with the decision boundary defined by theta.
    Plots the data points with * for the positive examples and o for the negative examples.

    """

    theta = np.array(theta)

    plot_data(x, y, possible_labels, feature_names)

    # following formula: order = (n^2 + 3n) / 2
    order = int(np.roots([1, 3, -2 * x.shape[1]])[1])

    u = np.linspace(x[:, 0].min(), x[:, 0].max(), 50)
    v = np.linspace(x[:, 1].min(), x[:, 1].max(), 50)

    z = np.zeros((u.size, v.size))
    # Evaluate z = theta*x over the grid
    for i, ui in enumerate(u):
        for j, vj in enumerate(v):
            z[i, j] = (
                feature_mapping(np.array([[ui, vj]]), order, intercept=True) @ theta
            )

    z = z.T

    # Plot z = 0
    plt.contour(u, v, z, levels=[0], linewidths=2, colors="r")
    plt.contourf(u, v, z, levels=[np.min(z), 0, np.max(z)], cmap="seismic", alpha=0.4)

    plt.legend(legend)


def plot_regression_line(theta, x, y, feature_names=None):
    """
    :param theta: parameters for logistic regression. A vector of shape (n+1, ).
    :param x: The input dataset. X is assumed to be a either:
            1) Mx2 matrix
            2) MxN, N > 2 matrix, where the N > 2 columns are the
                higher order terms of the first 2 features.
    :param y: vector of data labels of shape (m, ).
    :param feature_names: names of features, optional

    Plots the data points x and y into a new figure with the regression line defined by theta.

    """

    plot_data(x, y, feature_names=feature_names, logistic=False)

    # following formula: order = (n^2 + 3n) / 2
    order = x.shape[1]

    u = np.linspace(x[:, 0].min(), x[:, 0].max(), 50)

    v = np.zeros(u.size)
    # Evaluate v = theta*x over x
    for i, ui in enumerate(u):
        v[i] = feature_mapping(np.array([[ui]]), order, intercept=True) @ theta

    plt.plot(u, v, c="r")


def plot_costs(costs):
    """
    :param costs: 2-D array, consisting of the
    number of iterations and associated cost
    :return: None
    """

    plt.figure(figsize=(8, 8))

    plt.plot(costs[:, 0], costs[:, 1])

    plt.xlabel("Nr Iterations")
    plt.ylabel("J(\u03B8)")

    plt.show()
    

def plot_learning_curve(error_train, error_test, m_lst, optimize_lambda_=True):
    """
    Plots the learning curve for a given training and test sets.

    :param error_train: vector of training errors
    :param error_test: vector of test errors
    :param m_lst: vector of training examples used to calculate errors
    :param optimize_lambda_: if x axis is regularization parameter
    :return: None
    """

    plt.figure()
    
    plt.plot(m_lst, error_train, m_lst, error_test, lw=2)
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    if optimize_lambda_:
        plt.xlabel('lambda')
    else:
        plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()
