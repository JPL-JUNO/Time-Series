"""
@File         : 08_variance.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 17:51:04
@Email        : cuixuanstephen@gmail.com
@Description  : 异方差
"""

import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from scipy import stats

data = pd.read_csv(
    "assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]

series_daily = series.resample("D").sum()

series_df = series_daily.reset_index(drop=True).reset_index()
series_df.columns = ["time", "value"]
series_df["time"] += 1

olsr = ols("value ~ time", series_df).fit()

_, pval_white, _, _ = sms.het_white(olsr.resid, olsr.model.exog)
_, pval_bp, _, _ = sms.het_breuschpagan(olsr.resid, olsr.model.exog)

print(pval_white, pval_bp)


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(x) * (np.exp(np.abs(xt)) - 1)

        return x


series_log = LogTransformation.transform(series_daily)

# y = (x**lmbda - 1) / lmbda,  for lmbda != 0
# log(x),                  for lmbda = 0
series_transformed, lmbda = stats.boxcox(series_daily + 1)

series_transformed = pd.Series(series_transformed, index=series_daily.index)
