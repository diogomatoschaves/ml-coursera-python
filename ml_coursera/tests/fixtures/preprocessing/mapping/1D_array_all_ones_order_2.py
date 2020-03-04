import numpy as np

order = 2

base_array = np.array([1, 1, 1, 1]).reshape(-1, 1)

expected_array = np.ones(2 * base_array.size).reshape(-1, 2)
