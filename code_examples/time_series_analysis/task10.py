from sktime.datasets import load_shampoo_sales
from sktime.transformations.series.impute import Imputer
import pandas as pd
import numpy as np

y = load_shampoo_sales()
idx = y.index[:6]
y.loc[idx] = np.nan
imputer = Imputer(method="mean")
y_imputed = imputer.fit_transform(y)
pd.DataFrame({"imputed": y_imputed.values}, index=y_imputed.index).to_csv("outputs/task10.csv")

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task10.ttl")