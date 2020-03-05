import numpy as np
import pandas as pd
import os

from ml_coursera.regression import STRATEGY_OPTIONS

current_path = os.path.dirname(os.path.realpath(__file__))

strategy = "gradient_descent"

file_prefix = "multi"

file_specific = strategy if strategy in STRATEGY_OPTIONS else "gradient_descent"

data = pd.read_csv(
    os.path.join(current_path, f"{file_prefix}_feature_test_data.csv"), header=None
).to_numpy()

max_iter = 400

learning_rate = 0.1

normalize = True

expected_theta = np.array([[340412], [109447], [-6578]])

expected_predictions = (
    pd.read_csv(
        os.path.join(
            current_path, f"{file_prefix}_feature_{file_specific}_predictions.csv"
        ),
        header=None,
    )
    .to_numpy()
    .ravel()
)

expected_score = 0.732
