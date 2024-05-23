"""
@File         : 02_visualizing_timeseries.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 14:19:32
@Email        : cuixuanstephen@gmail.com
@Description  : 
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

series.plot(figsize=(12, 6), title="Solar radiation time series")

series_df = series.reset_index()
plt.rcParams["figure.figsize"] = [12, 6]
sns.set_theme(style="darkgrid")
# 使用 lineplot() 方法构建图表。
# 除了输入数据之外，它还采用每个轴的列名称：
# x 轴和 y 轴分别为 Datetime 和 Incoming Solar。
sns.lineplot(x="Datetime", y="Incoming Solar", data=series_df)

plt.ylabel("Solar Radiation")
plt.xlabel("")
plt.title("Solar radiation time series")
plt.savefig("./assets/time_series_plot.png")
