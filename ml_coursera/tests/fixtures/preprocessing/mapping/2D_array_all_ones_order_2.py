import numpy as np


order = 2

only_self_terms = False

base_array = np.array([[1, 1], [1, 1], [1, 1], [1, 1]])

size = (order ** 2 + 3 * order) / 2

expected_array = np.full((4, int(size)), 1)
