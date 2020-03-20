import numpy as np


order = 5

only_self_terms = False

base_array = np.array([[2, 3], [4, 1], [3, 2], [5, 4]])

expected_array = np.array(
    [
        [2, 3, 6, 4, 9, 18, 27, 8, 12, 24, 54, 81, 16, 36, 32, 72, 48, 243, 162, 108],
        [4, 1, 4, 16, 1, 4, 1, 64, 16, 64, 4, 1, 256, 16, 1024, 64, 256, 1, 4, 16],
        [3, 2, 6, 9, 4, 12, 8, 27, 18, 54, 24, 16, 81, 36, 243, 108, 162, 32, 48, 72],
        [
            5,
            4,
            20,
            25,
            16,
            80,
            64,
            125,
            100,
            500,
            320,
            256,
            625,
            400,
            3125,
            2000,
            2500,
            1024,
            1280,
            1600,
        ],
    ]
)
