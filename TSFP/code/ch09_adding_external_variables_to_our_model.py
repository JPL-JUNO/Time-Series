"""
@File         : ch09_adding_external_variables_to_our_model.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-04-05 14:01:41
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm
from typing import Union
import statsmodels.api as sm
from utils.optimize import optimize_SARIMAX
from itertools import product

warnings.filterwarnings("ignore")

macro_econ_data = sm.datasets.macrodata.load_pandas().data

fig, ax = plt.subplots()
ax.plot(macro_econ_data["realgdp"])
ax.set_xlabel("Date")
ax.set_ylabel("Real GDP (k$)")
plt.xticks(np.arange(0, 208, 16), np.arange(1959, 2010, 4))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH09_F01.png", dpi=300)

fig, axes = plt.subplots(3, 2, dpi=300, figsize=(11, 6))
for i, ax in enumerate(axes.flatten()):
    data = macro_econ_data[macro_econ_data.columns[i + 2]]
    ax.plot(data, color="black", linewidth=2)
    ax.set_title(macro_econ_data.columns[i + 1])
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    ax.spines["top"].set_alpha[0]
    ax.tick_params(labelsize=6)
plt.setp(axes, xticks=np.arange(0, 208, 8), xticklabels=np.arange(1959, 2010, 2))
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("figures/CH09_F02.png", dpi=300)

target = macro_econ_data["realgdp"]
exog = macro_econ_data[["realcons", "realinv", "realgovt", "realdpi", "cpi"]]
ad_fuller_result = adfuller(target)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

target_diff = np.diff(target)
ad_fuller_result = adfuller(target_diff)
print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 4
parameters = product(p, q, P, Q)
parameters_list = list(parameters)

target_train = target[:200]
exog_train = exog[:200]
result_df = optimize_SARIMAX(target_train, exog_train, parameters_list, d, D, s)

best_model = SARIMAX(
    target_train,
    exog_train,
    order=(3, 1, 3),
    seasonal_order=(0, 0, 0, 4),
    simple_differencing=False,
)
best_model_fit = best_model.fit(disp=False)
print(best_model_fit.summary())

best_model_fit.plot_diagnostics(figsize=(10, 8))
residuals = best_model_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))


def rolling_forecast(
    endog: Union[pd.Series, list],
    exog: Union[pd.Series, list],
    train_len: int,
    horizon: int,
    window: int,
    method: str,
) -> list:
    total_len = train_len + horizon
    if method == "last":
        pred_last_value = []
        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    elif method == "SARIMAX":
        pred_SARIMAX = []
        for i in range(train_len, total_len, window):
            model = SARIMAX(
                endog[:i],
                exog[:i],
                order=(3, 1, 3),
                seasonal_order=(0, 0, 0, 4),
                simple_differencing=False,
            )
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oss_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oss_pred)
        return pred_SARIMAX


target_train = target[:196]
target_test = target[196:]
pred_df = pd.DataFrame({"actual": target_test})
TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1
pred_last_value = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_SARIMAX = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, "SARIMAX")
pred_df["pred_last_value"] = pred_last_value
pred_df["pred_SARIMAX"] = pred_SARIMAX
