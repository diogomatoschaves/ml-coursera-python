import numpy as np
import pandas as pd
import os

from ml_coursera.preprocessing import feature_mapping

current_path = os.path.dirname(os.path.realpath(__file__))

file_prefix = "regularized"

data = pd.read_csv(
    os.path.join(current_path, f"2_features_{file_prefix}_test_data.csv"), header=None
).to_numpy()

data = np.c_[feature_mapping(data[:, :-1], 6), data[:, -1]]

max_iter = 800

learning_rate = 1

reg_param = 0.2

normalize = False

expected_theta = np.array([[ 2.253], [ 1.445], [ 2.332], [-2.350], [-3.570], [-3.070]])

expected_predictions = (
    pd.read_csv(
        os.path.join(current_path, f"2_features_{file_prefix}_predictions.csv"),
        header=None,
    )
    .to_numpy()
    .ravel()
)

expected_score = 0.8305
