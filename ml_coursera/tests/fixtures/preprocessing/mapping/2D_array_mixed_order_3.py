import numpy as np


order = 3

only_self_terms = False

base_array = np.array([[2, 3], [4, 1], [3, 2], [5, 4]])

expected_array = np.array(
    [
        [2, 3, 6, 4, 9, 18, 27, 8, 12],
        [4, 1, 4, 16, 1, 4, 1, 64, 16],
        [3, 2, 6, 9, 4, 12, 8, 27, 18],
        [5, 4, 20, 25, 16, 80, 64, 125, 100],
    ]
)
