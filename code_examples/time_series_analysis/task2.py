from sktime.datasets import load_shampoo_sales
import pandas as pd
import ruptures as rpt
import os

y = load_shampoo_sales().astype(float)
signal = y.to_numpy()

n_bkps = 3 
model = "l2"

algo = rpt.Binseg(model=model).fit(signal)
bkps = algo.predict(n_bkps=n_bkps) 
bkps = [b for b in bkps if b < len(y)] 

out = pd.DataFrame({
    "algorithm": ["binseg"] * len(bkps),
    "model": [model] * len(bkps),
    "n_bkps": [n_bkps] * len(bkps),
    "changepoint_pos": bkps, 
    "changepoint_time": [y.index[b] for b in bkps],
})
out.to_csv("outputs/task2.csv", index=False)

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task2.ttl")