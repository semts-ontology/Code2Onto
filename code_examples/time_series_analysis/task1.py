from sktime.datasets import load_airline
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.base import ForecastingHorizon
import pandas as pd

y = load_airline()
forecaster = ThetaForecaster(sp=12)
forecaster.fit(y)
fh = ForecastingHorizon(range(1, 13), is_relative=True)
forecast = forecaster.predict(fh)
pd.DataFrame({"forecast": forecast.values}, index=forecast.index).to_csv("outputs/task1.csv")

from ai_pipeline_orchestrator import analyze_code
analyze_code(output_path="tests/task1.ttl")