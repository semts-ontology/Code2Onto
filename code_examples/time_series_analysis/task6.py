from sktime.datasets import load_shampoo_sales, load_lynx
import pandas as pd
import numpy as np

x = load_shampoo_sales().astype(float)
z = load_lynx().astype(float)
m = min(len(x), len(z))
x_aligned = x.iloc[:m]
z_aligned = z.iloc[:m]
corr = float(np.corrcoef(x_aligned.values.reshape(-1), z_aligned.values.reshape(-1))[0, 1])
pd.DataFrame({"correlation": [corr]}).to_csv("outputs/task6.csv", index=False)

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task6.ttl")