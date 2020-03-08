import numpy as np
import pandas as pd
import os


current_path = os.path.dirname(os.path.realpath(__file__))

file_prefix = "unregularized"

data = pd.read_csv(
    os.path.join(current_path, f"2_features_{file_prefix}_test_data.csv"), header=None
).to_numpy()

reg_params = np.array([0, 0])

test_theta = np.array([[0, 0, 0], [-24, 0.2, 0.2]])

expected_cost = np.array([0.693, 0.218])

expected_gradient = np.array([[-0.1000, -12.0092, -11.2628], [0.043, 2.566, 2.647]])
