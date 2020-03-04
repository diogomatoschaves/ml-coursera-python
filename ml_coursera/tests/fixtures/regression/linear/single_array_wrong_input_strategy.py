import numpy as np
import pandas as pd
import os

from ml_coursera.regression import STRATEGY_OPTIONS

current_path = os.path.dirname(os.path.realpath(__file__))

strategy = "normal_equation"

file_specific = strategy if strategy in STRATEGY_OPTIONS else "gradient_descent"

data = pd.read_csv(
    os.path.join(current_path, "single_feature_test_data.csv"), header=None
).to_numpy()


max_iter = 1500

learning_rate = 0.01

expected_theta = np.array([[-3.630], [1.166]])

expected_predictions = (
    pd.read_csv(
        os.path.join(current_path, f"single_feature_{file_specific}_predictions.csv"),
        header=None,
    )
    .to_numpy()
    .ravel()
)
