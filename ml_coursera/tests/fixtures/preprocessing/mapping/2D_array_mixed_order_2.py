import numpy as np


order = 2

only_self_terms = False

base_array = np.array([[2, 3], [4, 1], [3, 2], [5, 4]])

expected_array = np.array(
    [[2, 3, 6, 4, 9], [4, 1, 4, 16, 1], [3, 2, 6, 9, 4], [5, 4, 20, 25, 16]]
)
