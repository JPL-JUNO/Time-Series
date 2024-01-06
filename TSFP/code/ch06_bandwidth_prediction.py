"""
@Title: 预测带宽数据
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-12-03 17:11:51
@Description: 
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.ts_plot import plot_pred
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from utils.arma import optimize_arma
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.rolling_forecast import rolling_forecast
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_csv('../data/bandwidth.csv')

fig, ax = plt.subplots()
ax.plot(df['hourly_bandwidth'])
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwith usage (MBps)')
plt.xticks(
    np.arange(0, 10000, 730),
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
     'Nov', 'Dec', '2020', 'Feb'])
fig.autofmt_xdate()
plt.tight_layout()

ADF_result = adfuller(df['hourly_bandwidth'])
print(f'ADF Statistics: {ADF_result[0]:.3f}')
print(f'p-value: {ADF_result[1]:.3f}')

bandwidth_diff = np.diff(df['hourly_bandwidth'], n=1)

ADF_result = adfuller(bandwidth_diff)
print(f'ADF Statistics: {ADF_result[0]:.3f}')
print(f'p-value: {ADF_result[1]:.3f}')

df_diff = pd.DataFrame({'bandwidth_diff': bandwidth_diff})
train = df_diff[:-168]
test = df_diff[-168:].copy()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
ax1.plot(df['hourly_bandwidth'])
ax1.set_xlabel('Time')
ax1.set_ylabel('Hourly bandwidth')
ax1.axvspan(9831, 10000, color='#808080', alpha=0.2)
ax2.plot(df_diff['bandwidth_diff'])
ax2.set_xlabel('Time')
ax2.set_ylabel('Hourly bandwidth (diff)')
ax2.axvspan(9830, 9999, color='#808080', alpha=0.2)
plt.xticks(
    np.arange(0, 10000, 730),
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', '2020', 'Feb'])
fig.autofmt_xdate()
plt.tight_layout()


ps = range(0, 4, 1)
qs = range(0, 4, 1)
order_list = list(product(ps, qs))
result_df = optimize_arma(train['bandwidth_diff'], order_list)

model = SARIMAX(train['bandwidth_diff'], order=(2, 0, 2),
                simple_differencing=False)
model_fit = model.fit(disp=False)

model_fit.plot_diagnostics(figsize=(10, 8))

print(model_fit.summary())

residuals = model_fit.resid
lb_df = acorr_ljungbox(residuals, np.arange(1, 11, 1))

# Forecasting bandwidth usage
pred_df = test.copy()

TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2
pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_arma = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'arma',
                             order=(2, 0, 2))

test.loc[:, 'pred_mean'] = pred_mean
test.loc[:, 'pred_last'] = pred_last
test.loc[:, 'pred_arma'] = pred_arma

fig, ax = plot_pred(test, ways=['mean', 'last', 'arma'])
ax.plot(df_diff['bandwidth_diff'])
ax.plot(test['bandwidth_diff'], label='actual')
ax.set_xlabel('Time')
ax.set_ylabel('Hourly bandwidth (diff)')
ax.axvspan(9830, 9999, color='#808080', alpha=0.2)
ax.set_xlim(9800, 9999)
ax.legend()
plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21'])
fig.autofmt_xdate()
plt.tight_layout()

mse_mean = mean_squared_error(test['bandwidth_diff'], test['pred_mean'])
mse_last = mean_squared_error(test['bandwidth_diff'], test['pred_last'])
mse_arma = mean_squared_error(test['bandwidth_diff'], test['pred_arma'])

print(mse_mean, mse_last, mse_arma)

df['pred_bandwidth'] = pd.Series()
df.loc[len(train)+1:, 'pred_bandwidth'] = (df['hourly_bandwidth'].iloc[len(train)+1] +
                                           test['pred_arma'].cumsum()).values

fig, ax = plt.subplots()
ax.plot(df['hourly_bandwidth'])
ax.plot(df['hourly_bandwidth'], 'b-', label='actual')
ax.plot(df['pred_bandwidth'], 'k--', label='ARMA(2,2)')
ax.axvspan(9831, 10000, color='#808080', alpha=0.2)
ax.set_xlim(9800, 9999)
plt.xticks(
    [9802, 9850, 9898, 9946, 9994],
    ['2020-02-13', '2020-02-15', '2020-02-17', '2020-02-19', '2020-02-21'])
fig.autofmt_xdate()
plt.tight_layout()

mae_diff_undiff = mean_absolute_error(df['hourly_bandwidth'][len(train)+1:],
                                      df['pred_bandwidth'][len(train)+1:])

print(mae_diff_undiff)
