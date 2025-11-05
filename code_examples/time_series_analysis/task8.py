from tslearn.datasets import UCR_UEA_datasets
from tslearn.neighbors import KNeighborsTimeSeriesClassifier
import pandas as pd

ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("ECG200")
clf = KNeighborsTimeSeriesClassifier(n_neighbors=1, metric="dtw")
clf.fit(X_train, y_train)
labels = clf.predict(X_test)
pd.DataFrame({"label": labels, "true": y_test}).to_csv("outputs/task8.csv", index=False)

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task8.ttl")