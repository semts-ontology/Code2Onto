from sktime.datasets import load_shampoo_sales
import numpy as np
import pandas as pd
import stumpy
import os

y = load_shampoo_sales().astype(float)
m = 6
mp = stumpy.stump(y.to_numpy(), m)
profile = mp[:, 0]
nn_idx = mp[:, 1].astype(int)

i = int(np.nanargmin(profile))
j = int(nn_idx[i])

out = pd.DataFrame({
    "window": [m],
    "start_a": [y.index[i]],
    "end_a": [y.index[i + m - 1]],
    "start_b": [y.index[j]],
    "end_b": [y.index[j + m - 1]],
    "distance": [profile[i]]
})

out.to_csv("outputs/task9.csv", index=False)

from remote_agent_server import analyze_code
analyze_code(output_path="tests/task9.ttl")