import numpy as np


feature_1 = np.array([1, 3, 2, 5]).reshape(-1, 1)
feature_2 = np.array([2, 1, 2, 10]).reshape(-1, 1)

base_array = np.c_[feature_1, feature_2]

expected_1 = np.array([-1.183, 0.169, -0.507, 1.521]).reshape(-1, 1)
expected_2 = np.array([-0.481, -0.757, -0.481, 1.721]).reshape(-1, 1)

expected_array = np.c_[expected_1, expected_2]
