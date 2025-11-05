from tslearn.datasets import UCR_UEA_datasets
from tslearn.metrics import dtw
import pandas as pd

ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("Trace")
seg1 = X_train[0].ravel()
seg2 = X_train[1].ravel()
d = float(dtw(seg1, seg2))
pd.DataFrame({"dtw_distance": [d]}).to_csv("outputs/task5.csv", index=False)

from remote_agent_server import analyze_code
analyze_code(output_path="tests/task5.ttl")