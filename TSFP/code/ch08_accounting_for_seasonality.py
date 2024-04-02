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
import matplotlib.pyplot as plt

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

test = df.iloc[-12:].copy()
test["naive_seasonal"] = df["Passengers"].iloc[120:132].values
ARIMA_pred = ARIMA_model_fit.get_prediction(132, 143).predicted_mean
test["ARIMA_pred"] = ARIMA_pred

### Forecasting with a SARIMA(p,d,q)(P,D,Q)m model
ad_fuller_result = adfuller(df["Passengers"])
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

df_diff = np.diff(df["Passengers"], n=1)
ad_fuller_result = adfuller(df_diff)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

df_diff_seasonal_diff = np.diff(df_diff, n=12)
ad_fuller_result = adfuller(df_diff_seasonal_diff)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

ps = range(0, 4, 1)
qs = range(0, 4, 1)
Ps = range(0, 4, 1)
Qs = range(0, 4, 1)
SARIMA_order_list = list(product(ps, qs, Ps, Qs))
train = df["Passengers"][:-12]

d = 1
D = 1
s = 12
SARIMA_result_df = optimize_SARIMA(train, SARIMA_order_list, d, D, s)

SARIMA_model = SARIMAX(
    train, order=(2, 1, 1), seasonal_order=(1, 1, 2, 12), simple_differencing=False
)
SARIMA_model_fit = SARIMA_model.fit(disp=False)
SARIMA_model_fit.plot_diagnostics(figsize=(10, 8))

residuals = SARIMA_model_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_result)

SARIMA_pred = SARIMA_model_fit.get_prediction(132, 143).predicted_mean
test["SARIMA_pred"] = SARIMA_pred

### Comparing the performance of each forecasting method
fig, ax = plt.subplots()
ax.plot(df["Month"], df["Passengers"])
ax.plot(test["Passengers"], "b-", label="actual")
ax.plot(test["naive_seasonal"], "r:", label="naive seasonal")
ax.plot(test["ARIMA_pred"], "k--", label="ARIMA(11,2,3)")
ax.plot(test["SARIMA_pred"], "g-.", label="SARIMA(2,1,1)(1,1,2,12)")
ax.set_xlabel("Date")
ax.set_ylabel("Number of air passengers")
ax.axvspan(132, 143, color="#808080", alpha=0.2)
ax.legend()
plt.xticks(np.arange(0, 145, 12), np.arange(1949, 1962, 1))
ax.set_xlim(120, 143)
fig.autofmt_xdate()
plt.tight_layout()

from sklearn.metrics import mean_absolute_percentage_error

mape_naive_seasonal = 100 * mean_absolute_percentage_error(
    test["Passengers"], test["naive_seasonal"]
)
mape_ARIMA = 100 * mean_absolute_percentage_error(
    test["Passengers"], test["ARIMA_pred"]
)
mape_SARIMA = 100 * mean_absolute_percentage_error(
    test["Passengers"], test["SARIMA_pred"]
)
fig, ax = plt.subplots()
x = ["naive seasonal", "ARIMA(11,2,3)", "SARIMA(2,1,1)(1,1,2,12)"]
y = [mape_naive_seasonal, mape_ARIMA, mape_SARIMA]
ax.bar(x, y, width=0.4)
ax.set_xlabel("Models")
ax.set_ylabel("MAPE (%)")
ax.set_ylim(0, 15)
for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(round(value, 2)), ha="center")
plt.tight_layout()
