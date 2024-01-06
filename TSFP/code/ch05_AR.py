"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-12-01 19:18:28
@Description: 
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.ts_plot import plot_pred
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from utils.rolling_forecast import rolling_forecast

df = pd.read_csv("../data/foot_traffic.csv")

fig, ax = plt.subplots()
ax.plot(df["foot_traffic"])
ax.set_xlabel("Time")
ax.set_ylabel("Average weekly foot traffic")
plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.tight_layout()


ADF_result = adfuller(df["foot_traffic"])

print(f"ADF Statistics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")

foot_traffic_diff = np.diff(df["foot_traffic"], n=1)
ADF_result = adfuller(foot_traffic_diff)

print(f"ADF Statistics: {ADF_result[0]}")
print(f"p-value: {ADF_result[1]}")


fig = plot_acf(foot_traffic_diff, lags=20)
plt.tight_layout()


# PACF

np.random.seed(42)
ma2 = np.array([1, 0, 0])
ar2 = np.array([1, -0.33, -0.5])
AR2_process = ArmaProcess(ar2, ma2).generate_sample(nsample=1_000)

fig = plot_pacf(AR2_process, lags=20)
plt.tight_layout()

fig = plot_pacf(foot_traffic_diff, lags=20)
plt.tight_layout()

df_diff = pd.DataFrame({'foot_traffic_diff': foot_traffic_diff})
train = df_diff[:-52]
test = df_diff[-52:].copy()


fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
ax1.plot(df['foot_traffic'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Avg. weekly foot traffic')
ax1.axvspan(948, 1000, color='#808080', alpha=0.2)
ax2.plot(df_diff['foot_traffic_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Diff. avg. weekly foot traffic')
ax2.axvspan(947, 999, color='#808080', alpha=0.2)
plt.xticks(np.arange(0, 1000, 104), np.arange(2000, 2020, 2))
fig.autofmt_xdate()
plt.tight_layout()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 1

pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_ar = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'AR')

test.loc[:, 'pred_mean'] = pred_mean
test.loc[:, 'pred_last'] = pred_last
test.loc[:, 'pred_ar'] = pred_ar

fig, ax = plot_pred(test, ways=['mean', 'last', 'ar'])
ax.plot(test['foot_traffic_diff'], 'b-', label='actual')
ax.plot(df_diff['foot_traffic_diff'])
ax.set_xlabel('Time')
ax.set_ylabel('Diff. avg. weekly foot traffic')
ax.axvspan(947, 998, color='#808080', alpha=0.2)
ax.set_xlim(920, 999)
plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()


mse_mean = mean_squared_error(test['foot_traffic_diff'], test['pred_mean'])
mse_last = mean_squared_error(test['foot_traffic_diff'], test['pred_last'])
mse_AR = mean_squared_error(test['foot_traffic_diff'], test['pred_ar'])

print(mse_mean, mse_last, mse_AR)

df['pred_foot_traffic'] = pd.Series()
df.loc[len(train)+1:, 'pred_foot_traffic'] = (df['foot_traffic'].iloc[len(
    train)+1]+test['pred_ar'].cumsum()).values

fig, ax = plt.subplots()
ax.plot(df['foot_traffic'])
ax.plot(df['foot_traffic'], 'b-', label='actual')
ax.plot(df['pred_foot_traffic'], 'k--', label='AR(3)')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Average weekly foot traffic')
ax.axvspan(948, 1000, color='#808080', alpha=0.2)
ax.set_xlim(920, 1000)
ax.set_ylim(650, 770)
plt.xticks([936, 988], [2018, 2019])
fig.autofmt_xdate()
plt.tight_layout()

mae_ar_undiff = mean_absolute_error(df['foot_traffic'][948:],
                                    df['pred_foot_traffic'][948:])
