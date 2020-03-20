import numpy as np

order = 5

only_self_terms = True

base_array = np.array([[2, 3], [4, 1], [3, 2], [5, 4]])

expected_array = np.array([
    [2,  3,  4,  9,  8, 27, 16, 81, 32,  243],
    [4,  1, 16,  1, 64,  1,  256,  1, 1024,  1],
    [3,  2,  9,  4, 27,  8, 81, 16,  243, 32],
    [5,  4, 25, 16,  125, 64,  625,  256, 3125, 1024]]
)
