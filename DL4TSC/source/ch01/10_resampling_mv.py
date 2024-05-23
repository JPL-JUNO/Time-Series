"""
@File         : 10_resampling_mv.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 21:23:30
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import pandas as pd
import numpy as np

data = pd.read_csv(
    "assets/data/time_series_smf1.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

stat_by_variable = {
    "Incoming Solar": "sum",
    "Wind Dir": "mean",
    "Snow Depth": "sum",
    "Wind Speed": "mean",
    "Dewpoint": "mean",
    "Precipitation": "sum",
    "Vapor Pressure": "mean",
    "Relative Humidity": "mean",
    "Air Temp": "max",
}

data_daily = data.resample("D").agg(stat_by_variable)

data_daily.iloc[-365:].plot(figsize=(15, 6))

data_logscale = np.sign(data_daily) * np.log(np.abs(data_daily) + 1)
data_logscale.plot(figsize=(15, 8))
data_logscale.iloc[: 365 * 2].plot(figsize=(15, 6))
