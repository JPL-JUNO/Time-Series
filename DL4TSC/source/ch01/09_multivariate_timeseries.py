"""
@File         : 09_multivariate_timeseries.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 18:21:51
@Email        : cuixuanstephen@gmail.com
@Description  : 加载并可视化多元时间序列
"""

import pandas as pd
import numpy as np
import matplotlib

data = pd.read_csv(
    "assets/data/time_series_smf1.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


data_log = LogTransformation.transform(data)

mv_plot = data_log.iloc[-1_000:].plot(
    figsize=(15, 8), title="Multivariate time series", xlabel="", ylabel="Value"
)
mv_plot.legend(fancybox=True, framealpha=1)
mv_plot.figure.savefig("assets/mts_plot.png")
