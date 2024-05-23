"""
@File         : 01_loading_timeseries.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 14:15:28
@Email        : cuixuanstephen@gmail.com
@Description  : 使用 pandas 加载时间序列
"""

import pandas as pd

data = pd.read_csv(
    "./assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]

# series = pd.Series(data=data["Incoming Solar"], index=data.index)
