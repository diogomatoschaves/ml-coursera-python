import numpy as np


order = 3

only_self_terms = False

base_array = np.array([2, 4, 3, 5]).reshape(-1, 1)

expected_array = np.array([[2, 4, 8], [4, 16, 64], [3, 9, 27], [5, 25, 125]])
