"""
@File         : ch08_accounting_for_seasonality.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-03-31 13:47:19
@Email        : cuixuanstephen@gmail.com
@Description  : 考虑季节性
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from itertools import product
from typing import Union
from tqdm.notebook import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("../data/air-passengers.csv")
ad_fuller_result = adfuller(df["Passengers"])
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value :{ad_fuller_result[1]}")

df_diff = np.diff(df["Passengers"], n=1)
ad_fuller_result = adfuller(df_diff)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value :{ad_fuller_result[1]}")

df_diff2 = np.diff(df_diff, n=1)
ad_fuller_result = adfuller(df_diff2)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value :{ad_fuller_result[1]}")


def adf(ts, d=0):
    result = adfuller(ts)
    if result[1] < 0.05:
        return d
    d += 1
    return adf(np.diff(ts), d)


ps = range(0, 13, 1)
qs = range(0, 13, 1)
# Set P and Q to 0, since we are working with an ARIMA(p,d,q) model.
Ps = [0]
Qs = [0]
d = 2
D = 0
# The parameter s is equivalent to m. They both denote the frequency.
s = 12
ARIMA_order_list = list(product(ps, qs, Ps, Qs))


# We set P, D, and Q to 0 because a
# SARIMA(p,d,q)(0,0,0)m model is equivalent to an ARIMA(p,d,q) model.
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


train = df["Passengers"][:-12]
ARIMA_result_df = optimize_SARIMA(train, ARIMA_order_list, d, D, s)
print(ARIMA_result_df)  # SARIMA(11,2,3)(0,0,0)12 model

ARIMA_model = SARIMAX(train, order=(11, 2, 3), simple_differencing=False)
ARIMA_model_fit = ARIMA_model.fit(disp=False)
ARIMA_model_fit.plot_diagnostics(figsize=(10, 8))

from statsmodels.stats.diagnostic import acorr_ljungbox

residuals = ARIMA_model_fit.resid
lb_test_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
