from sktime.datasets import load_lynx
import pandas as pd
import numpy as np

y = load_lynx()
n_train = int(0.7 * len(y))
train, test = y.iloc[:n_train], y.iloc[n_train:]
train_mean = float(train.mean())
train_std = float(train.std())
z_test = (test - train_mean) / (train_std if train_std != 0 else 1.0)
anomalies = (np.abs(z_test) > 2.0).astype(int)
res = pd.DataFrame({"test_value": test.values, "zscore": z_test.values, "anomalies": anomalies.values, "train_mean": train_mean, "train_std": train_std}, index=test.index)
res.to_csv("outputs/task4.csv")

from remote_agent_server import analyze_code
analyze_code(output_path="tests/task4.ttl")