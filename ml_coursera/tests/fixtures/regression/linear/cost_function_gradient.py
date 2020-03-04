import numpy as np
import pandas as pd
import os


current_path = os.path.dirname(os.path.realpath(__file__))

data = pd.read_csv(
    os.path.join(current_path, "single_feature_test_data.csv"), header=None
).to_numpy()

theta_1 = np.array([0, 0])
theta_2 = np.array([-1, 2])

expected_cost_1 = 32.07
expected_cost_2 = 54.24
