"""
@File         : 05_decomposition.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 16:17:06
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

data = pd.read_csv(
    "./assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"]
series_daily = series.resample("D").sum()

result = seasonal_decompose(x=series_daily, model="additive", period=365)

plot = result.plot()
plt.tight_layout()
plot.savefig("assets/classical_decomposition.png")

# result.trend, result.seasonal, result.resid

from statsmodels.tsa.seasonal import STL

result = STL(endog=series_daily, period=365).fit()
plot = result.plot()

# MSTL
from statsmodels.tsa.seasonal import MSTL

result = MSTL(endog=series_daily, periods=(7, 365)).fit()
plot = result.plot()
