"""
@File         : 03_resampling.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 14:28:15
@Email        : cuixuanstephen@gmail.com
@Description  : 重新采样时间序列
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    "./assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]

series_daily = series.resample("D").sum()

series_df = series_daily.reset_index()

plt.rcParams["figure.figsize"] = [12, 6]

sns.set_theme(style="darkgrid")

sns.lineplot(x="Datetime", y="Incoming Solar", data=series_df)

plt.ylabel("Solar Radiation")
plt.xlabel("")
plt.title("Daily total solar radiation")

plt.savefig("assets/daily_time_series_plot.png")
