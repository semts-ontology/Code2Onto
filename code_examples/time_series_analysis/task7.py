from tslearn.datasets import UCR_UEA_datasets
from tslearn.clustering import TimeSeriesKMeans
import pandas as pd
import numpy as np

ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("Trace")
n_subset = 20
X_small = X_train[:n_subset]
y_small = y_train[:n_subset]
kmeans = TimeSeriesKMeans(n_clusters=3, metric="dtw", random_state=0)
labels = kmeans.fit_predict(X_small)
pd.DataFrame({"label": labels, "target": y_small}).to_csv("outputs/task7.csv", index=False)

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task7.ttl")