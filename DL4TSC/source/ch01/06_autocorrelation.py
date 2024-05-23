"""
@File         : 06_autocorrelation.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 16:47:19
@Email        : cuixuanstephen@gmail.com
@Description  : 自相关
"""

import pandas as pd
import matplotlib
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# matplotlib.use("TkAgg")

data = pd.read_csv(
    "assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]

series_daily = series.resample("D").sum()

acf_scores = acf(x=series_daily, nlags=365)
pacf_scores = pacf(x=series_daily, nlags=365)

acf_plot_ = plot_acf(series_daily, lags=365)
pacf_plot_ = plot_pacf(series_daily, lags=365)
# plot = plot_acf(series, lag)
