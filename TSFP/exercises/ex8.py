"""
@File         : ex8.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-04-02 22:03:25
@Email        : cuixuanstephen@gmail.com
@Description  : Apply the SARIMA(p,d,q)(P,D,Q)m model on the Johnson & Johnson dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller
from typing import Union
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import warnings

warnings.filterwarnings("ignore")


df = pd.read_csv("../data/jj.csv")
fig, ax = plt.subplots()
ax.plot(df.date, df["data"])
ax.set_xlabel("Date")
ax.set_ylabel("Earnings per share (USD)")
plt.xticks(
    np.arange(0, 81, 8),
    [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980],
)
fig.autofmt_xdate()
plt.tight_layout()

decomposition = STL(
    df["data"], period=4
).fit()  # Period is 4 since we have quarterly data

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 8))
axes[0].plot(decomposition.observed)
axes[0].set_ylabel("Observed")

axes[1].plot(decomposition.trend)
axes[1].set_ylabel("Trend")

axes[2].plot(decomposition.seasonal)
axes[2].set_ylabel("Seasonal")

axes[3].plot(decomposition.resid)
axes[3].set_ylabel("Residuals")

plt.xticks(
    np.arange(0, 81, 8),
    [1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980],
)

fig.autofmt_xdate()
plt.tight_layout()

ad_fuller_result = adfuller(df["data"])
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

df_diff = np.diff(df["data"], n=1)
ad_fuller_result = adfuller(df_diff)

print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

# 尝试季节性差分
df_diff_seasonal_diff = np.diff(df_diff, n=4)
ad_fuller_result = adfuller(df_diff_seasonal_diff)

print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")


def optimize_SARIMA(
    endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int
) -> pd.DataFrame:
    results = []
    for order in tqdm(order_list):
        try:
            model = SARIMAX(
                endog,
                order=(order[0], d, order[1]),
                seasonal_order=(order[2], D, order[3], s),
                simple_differencing=False,
            ).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q,P,Q)", "AIC"]

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


train = df[:-4]
test = df[-4:].copy()
ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)

SARIMA_order_list = list(product(ps, qs, Ps, Qs))

d = 1
D = 1
s = 4  # We have quarterly data, so 4 data points per seasonal cycle

SARIMA_result_df = optimize_SARIMA(train["data"], SARIMA_order_list, d, D, s)
SARIMA_result_df
