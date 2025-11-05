from sktime.datasets import load_shampoo_sales
import pandas as pd
import numpy as np

y = load_shampoo_sales()
mean_val = float(y.mean())
std_val = float(y.std())
z = (y - mean_val) / (std_val if std_val != 0 else 1.0)
anomalies = (np.abs(z) > 2.0).astype(int)
res = pd.DataFrame({"value": y.values, "zscore": z.values, "anomalies": anomalies.values, "mean": mean_val, "std": std_val}, index=y.index)
res.to_csv("outputs/task3.csv")

from remote_agent_server import analyze_code
analyze_code(output_path="tests/task3.ttl")