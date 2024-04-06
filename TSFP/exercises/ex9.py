"""
@File         : ex9.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-04-06 18:50:09
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

# 1. Use all exogenous variables in the SARIMAX model.
# 2. Perform residual analysis.
# 3. Produce forecasts for the last seven timesteps in the dataset.
# 4. Measure the MAPE. Is it better, worse, or identical to what was achieved with a limited number of exogenous variables?

import warnings
import sys
from os import path

sys.path.append(path.normpath(path.join(path.dirname(__file__), "../")))
from utils.optimize import optimize_SARIMAX
from utils.forecast import rolling_forecast
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from itertools import product
from sklearn.metrics import mean_absolute_percentage_error

warnings.filterwarnings("ignore")

macro_econ_data = sm.datasets.macrodata.load_pandas().data
target = macro_econ_data["realgdp"]
exog = macro_econ_data[
    [
        "realcons",
        "realinv",
        "realgovt",
        "realdpi",
        "cpi",
        "m1",
        "tbilrate",
        "unemp",
        "pop",
        "infl",
        "realint",
    ]
]
ad_fuller_result = adfuller(target)

print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

target_diff = np.diff(target, n=1)
ad_fuller_result = adfuller(target_diff[1:])

print(f"ADF Statistic: {ad_fuller_result[0]}")
print(f"p-value: {ad_fuller_result[1]}")

target_train = target[:200]
exog_train = exog[:200]
p = range(0, 4, 1)
d = 1
q = range(0, 4, 1)
P = range(0, 4, 1)
D = 0
Q = range(0, 4, 1)
s = 4

parameters = product(p, q, P, Q)
parameters_list = list(parameters)
result_df = optimize_SARIMAX(target, exog, parameters_list, d, D, s)
print(result_df)

best_model = SARIMAX(
    target_train,
    exog_train,
    order=(2, 1, 2),
    seasonal_order=(1, 0, 0, 4),
    simple_differencing=False,
)
best_model_fit = best_model.fit(disp=False)
print(best_model_fit.summary())

best_model_fit.plot_diagnostics(figsize=(10, 8))
residuals = best_model_fit.resid
lb_result = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_result)

target_train = target[:196]
target_test = target[:196]
pred_df = pd.DataFrame({"actual": target_test})
TRAIN_LEN = len(target_train)
HORIZON = len(target_test)
WINDOW = 1

pred_last_value = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, "last")
pred_SARIMAX = rolling_forecast(target, exog, TRAIN_LEN, HORIZON, WINDOW, "SARIMAX")
pred_df["pred_last_value"] = pred_last_value
pred_df["pred_SARIMAX"] = pred_SARIMAX

mape_last = 100 * mean_absolute_percentage_error(
    pred_df.actual, pred_df.pred_last_value
)
mape_SARIMAX = 100 * mean_absolute_percentage_error(
    pred_df.actual, pred_df.pred_SARIMAX
)
fig, ax = plt.subplots()

x = ["naive last value", "SARIMAX"]
y = [mape_last, mape_SARIMAX]

ax.bar(x, y, width=0.4)
ax.set_xlabel("Models")
ax.set_ylabel("MAPE (%)")
ax.set_ylim(0, 1)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 0.05, s=str(round(value, 2)), ha="center")

plt.tight_layout()
