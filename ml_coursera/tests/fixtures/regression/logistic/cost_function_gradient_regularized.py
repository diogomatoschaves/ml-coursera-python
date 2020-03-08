import numpy as np
import pandas as pd
import os
from ml_coursera.preprocessing import feature_mapping


current_path = os.path.dirname(os.path.realpath(__file__))

file_prefix = "regularized"

data = pd.read_csv(
    os.path.join(current_path, f"2_features_{file_prefix}_test_data.csv"), header=None
).to_numpy()

data = np.c_[feature_mapping(data[:, :-1], 6), data[:, -1]]

reg_params = [1, 10]

test_theta = np.array([
    np.zeros(data.shape[1]), # number of features including polynomial terms and intercept
    np.ones(data.shape[1])
])

expected_cost = np.array([0.693, 3.16])

expected_gradient = np.array([
    [0.0085, 0.0188, 0.0001, 0.0503, 0.0115],
    [0.3460, 0.1614, 0.1948, 0.0922, 0.2269]
])
