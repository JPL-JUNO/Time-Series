"""
@File         : 07_stationarity.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 17:30:15
@Email        : cuixuanstephen@gmail.com
@Description  : 检测平稳性
"""

import pandas as pd
from pmdarima.arima import ndiffs, nsdiffs
import matplotlib.pyplot as plt


data = pd.read_csv(
    "assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]

series_daily = series.resample("D").sum()

ndiffs(x=series_daily, test="kpss")
ndiffs(x=series_daily, test="adf")

series_changes = series_daily.diff()[1:]
plot = series_changes.plot(title="Changes in solar radiation in consecutive days")
plot.figure.savefig("assets/uts_daily_changes.png")

# 不是很理解，这里还有必要进行季节性的差分吗？
# 原本的差分后的数据已经平稳了
nsdiffs(x=series_changes, test="ch", m=365)
nsdiffs(x=series_changes, test="ocsb", m=365)
