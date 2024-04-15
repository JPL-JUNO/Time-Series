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
