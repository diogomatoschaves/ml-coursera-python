import numpy as np
import pandas as pd
import os


current_path = os.path.dirname(os.path.realpath(__file__))

file_prefix = "unregularized"

data = pd.read_csv(
    os.path.join(current_path, f"2_features_{file_prefix}_test_data.csv"), header=None
).to_numpy()

max_iter = 400

learning_rate = 1

reg_param = 0

normalize = True

expected_theta = np.array([[1.659], [3.867], [3.603]])

expected_predictions = (
    pd.read_csv(
        os.path.join(current_path, f"2_features_{file_prefix}_predictions.csv"),
        header=None,
    )
    .to_numpy()
    .ravel()
)

expected_score = 0.89
