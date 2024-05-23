"""
@File         : 11_correlation.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 21:36:44
@Email        : cuixuanstephen@gmail.com
@Description  : 分析变量对之间的相关性
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# calculate the correlation matrix
corr_matrix = data_daily.corr(method="pearson")

sns.heatmap(
    data=corr_matrix,
    cmap=sns.diverging_palette(230, 20, as_cmap=True),
    xticklabels=data_daily.columns,
    yticklabels=data_daily.columns,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)

plt.xticks(rotation=60)
plt.tick_params(axis="both", which="both", length=0)

plt.savefig("assets/corr_heatmap.png")
