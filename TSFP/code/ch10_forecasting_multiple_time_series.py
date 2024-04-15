"""
@File         : ch10_forecasting_multiple_time_series.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-04-14 18:07:27
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from utils.print_info import print_adf_result
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import pandas as pd

warnings.filterwarnings("ignore")


def print_adf_result(res, variable):
    print(variable)
    print(f"ADF Statistic: {res[0]}")
    print(f"p-value: {res[1]}")


macro_econ_data = sm.datasets.macrodata.load_pandas().data

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
ax1.plot(macro_econ_data["realdpi"])
ax1.set_xlabel("Date")
ax1.set_ylabel("Real disposable income (k$)")
ax1.set_title("realdpi")
ax1.spines["top"].set_alpha(0)
ax2.plot(macro_econ_data["realcons"])
ax2.set_xlabel("Date")
ax2.set_ylabel("Real consumption (k$)")
ax2.set_title("realcons")
ax2.spines["top"].set_alpha(0)
plt.xticks(np.arange(0, 208, 16), np.arange(1959, 2010, 4))
fig.autofmt_xdate()
plt.tight_layout()

# BUG 是不是应该先分 train 和 test，在进行差分的判断？
ad_fuller_result_1 = adfuller(macro_econ_data["realdpi"])
print_adf_result(ad_fuller_result_1, "realdpi")

print("\n---------------------\n")

ad_fuller_result_2 = adfuller(macro_econ_data["realcons"])
print_adf_result(ad_fuller_result_2, "realcons")

ad_fuller_result_1 = np.diff(macro_econ_data["realdpi"], n=1)
print_adf_result(ad_fuller_result_1, "realcons")
ad_fuller_result_2 = np.diff(macro_econ_data["realcons"], n=1)
print_adf_result(ad_fuller_result_2, "realcons")

from utils.optimize import optimize_VAR

endog = macro_econ_data[["realdpi", "realcons"]]
endog_diff = endog.diff()[1:]

train = endog_diff[:162]
test = endog_diff[162:]

result_df = optimize_VAR(train)

print("realcons Granger-causes realdpi?\n")
print("------------------")
granger_1 = grangercausalitytests(
    macro_econ_data[["realdpi", "realcons"]].diff()[1:], 3
)

print("\nrealdpi Granger-causes realcons?\n")
print("------------------")
granger_2 = grangercausalitytests(
    macro_econ_data[["realcons", "realdpi"]].diff()[1:], [3]
)

# Running the Granger causality test for both variables returns a p-value smaller than
# 0.05 in both cases. Therefore, we can reject the null hypothesis and conclude that
# realdpi Granger-causes realcons, and realcons Granger-causes realdpi.

best_model = VARMAX(train, order=(3, 0))
best_model_fit = best_model.fit(disp=False)
print(best_model_fit.summary())

best_model_fit.plot_diagnostics(figsize=(10, 7), variable=0)
plt.savefig("img/CH10_F04.png", dpi=300)
# realcons
best_model_fit.plot_diagnostics(figsize=(10, 8), variable=1)
plt.savefig("img/CH10_F05.png", dpi=300)

realgdp_residuals = best_model_fit.resid["realdpi"]
lb_result = acorr_ljungbox(realgdp_residuals, np.arange(1, 11, 1))
print(lb_result)
realcons_residuals = best_model_fit.resid["realcons"]
lb_result = acorr_ljungbox(realcons_residuals, np.arange(1, 11, 1))
print(lb_result)

from utils.forecast import rolling_forecast_varmax

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 4

realdpi_pred_VAR, realcons_pred_VAR = rolling_forecast_varmax(
    endog_diff,
    TRAIN_LEN,
    HORIZON,
    WINDOW,
    "VAR",
    order=(3, 0),  # 注意这里没有传入差分的参数 d，因此后面的预测需要手动的累加起来
    # VAR 需要序列是平稳的，0 表示的是参数 q
)

test = endog[163:].copy()
# REMOVE
test["realdpi_pred_VAR"] = pd.Series()  # 增加一列全是 NaN 的
test["realdpi_pred_VAR"] = endog.iloc[162]["realdpi"] + np.cumsum(realdpi_pred_VAR)
test["realcons_pred_VAR"] = pd.Series()
test["realcons_pred_VAR"] = endog.iloc[162]["realcons"] + np.cumsum(realcons_pred_VAR)

realdpi_pred_baseline, realcons_pred_baseline = rolling_forecast_varmax(
    endog, TRAIN_LEN, HORIZON, WINDOW, "last", order=(3, 0)
)
test["realdpi_pred_baseline"] = realdpi_pred_baseline
test["realcons_pred_baseline"] = realcons_pred_baseline

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
ax1.plot(macro_econ_data["realdpi"])
ax1.plot(test["realdpi_pred_VAR"], "k--", label="VAR")
ax1.plot(test["realdpi_pred_baseline"], "r:", label="Last(baseline)")
ax1.set_xlabel("Date")
ax1.set_ylabel("Real disposable income ($)")
ax1.set_title("realdpi")
ax1.spines["top"].set_alpha(0)
ax1.axvspan(163, 202, color="#808080", alpha=0.2)
ax1.set_xlim(100, 202)
ax1.legend()
ax2.plot(macro_econ_data["realcons"])
ax2.plot(test["realcons_pred_VAR"], "k--", label="VAR")
ax2.plot(test["realcons_pred_baseline"], "r:", label="Last(baseline)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Real consumption (k$)")
ax2.set_title("realcons")
ax2.spines["top"].set_alpha(0)
ax2.axvspan(163, 202, color="#808080", alpha=0.2)
ax2.set_xlim(100, 202)
ax2.legend(loc=2)
plt.xticks(np.arange(0, 208, 16), np.arange(1959, 2010, 4))
plt.xlim(100, 202)
fig.autofmt_xdate()
plt.tight_layout()
plt.savefig("img/CH10_F06.png", dpi=300)

from sklearn.metrics import mean_absolute_percentage_error

mape_realdpi_VAR = (
    mean_absolute_percentage_error(test["realdpi"], test["realdpi_pred_VAR"]) * 100
)
mape_realcons_VAR = (
    mean_absolute_percentage_error(test["realcons"], test["realcons_pred_VAR"]) * 100
)
mape_realdpi_baseline = (
    mean_absolute_percentage_error(test["realcons"], test["realcons_pred_baseline"])
    * 100
)
mape_realcons_baseline = (
    mean_absolute_percentage_error(test["realcons"], test["realcons_pred_baseline"])
    * 100
)

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
x = ["baseline", "VAR"]
y1 = [mape_realdpi_baseline, mape_realdpi_VAR]
y2 = [mape_realcons_baseline, mape_realcons_VAR]
ax1.bar(x, y1)
ax1.set_xlabel("Methods")
ax1.set_ylabel("MAPE(%)")
ax1.set_title("realdpi")
ax1.set_ylim(0, 3.5)

ax2.bar(x, y2)
ax2.set_xlabel("Methods")
ax2.set_ylabel("MAPE (%)")
ax2.set_title("realcons")
ax2.set_ylim(0, 3)

for index, value in enumerate(y1):
    ax1.text(x=index, y=value + 0.05, s=str(round(value, 2)), ha="center")
for index, value in enumerate(y2):
    ax1.text(x=index, y=value + 0.05, s=str(round(value, 2)), ha="center")
plt.tight_layout()
plt.savefig("img/CH10_F07.png", dpi=300)
