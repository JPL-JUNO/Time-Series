"""
@Title: MA 建模
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-29 21:56:20
@Description: 
"""

from utils.ts_plot import plot_metrics_compare
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.ts_plot import plot_pred
from utils.rolling_forecast import rolling_forecast
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('../data/widget_sales.csv')

fig, ax = plt.subplots()
ax.plot(df['widget_sales'])
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (k$)')
plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409,
     439, 468, 498],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
     'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()


ADF_result = adfuller(df['widget_sales'])
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')


widget_sales_diff = np.diff(df['widget_sales'], n=1)

ADF_result = adfuller(widget_sales_diff)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

fig = plot_acf(widget_sales_diff, lags=30)
plt.tight_layout()


df_diff = pd.DataFrame({'widget_sales_diff': widget_sales_diff})

train_size = int(.9*len(df_diff))
train = df_diff[:train_size].copy()
test = df_diff[train_size:].copy()

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
ax1.plot(df['widget_sales'])
ax1.axvspan(train_size, len(df), color='#808080', alpha=.2)
ax2.plot(df_diff['widget_sales_diff'])
ax2.axvspan(train_size-1, len(df)-1, color='#808080', alpha=.2)
plt.xticks(
    [0, 30, 57, 87, 116, 145, 175, 204, 234, 264, 293, 323, 352, 382, 409,
     439, 468, 498],
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
     'Nov', 'Dec', '2020', 'Feb', 'Mar', 'Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()

pred_df = test.copy()
TRAIN_LEN = len(train)
HORIZON = len(test)
WINDOW = 2
pred_mean = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'mean')
pred_last = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'last')
pred_ma = rolling_forecast(df_diff, TRAIN_LEN, HORIZON, WINDOW, 'MA')

pred_df.loc[:, 'pred_mean'] = pred_mean
pred_df.loc[:, 'pred_last'] = pred_last
pred_df.loc[:, 'pred_MA'] = pred_ma


fig, ax = plot_pred(pred_df, ways=['mean', 'last', 'MA'])
ax.plot(df_diff['widget_sales_diff'], label='actual')
ax.legend()
ax.axvspan(449, 498, color='#808080', alpha=0.2)
ax.set_xlim(430, 500)
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales - diff (k$)')
plt.xticks(
    [439, 468, 498],
    ['Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()

mse_mean = mean_squared_error(
    pred_df['widget_sales_diff'], pred_df['pred_mean'])
mse_last = mean_squared_error(
    pred_df['widget_sales_diff'], pred_df['pred_last'])
mse_ma = mean_squared_error(pred_df['widget_sales_diff'], pred_df['pred_MA'])

fig, ax = plot_metrics_compare(['mean', 'last', 'MA'],
                               [mse_mean, mse_last, mse_ma])


df['pred_widgets_sales'] = pd.Series()
# 得使用 values，否则的话会被 index alignment
df.loc[450:, 'pred_widgets_sales'] = (df['widget_sales'].iloc[450] +
                                      pred_df['pred_MA'].cumsum()).values

fig, ax = plt.subplots()
ax.plot(df['widget_sales'], 'b-', label='actual')
ax.plot(df['pred_widgets_sales'], 'k--', label='MA(2)')
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Widget sales (K$)')
ax.axvspan(450, 500, color='#808080', alpha=.2)
ax.set_xlim(400, 500)
plt.xticks(
    [409, 439, 468, 498],
    ['Mar', 'Apr', 'May', 'Jun'])
fig.autofmt_xdate()
plt.tight_layout()

mae_ma_undiff = mean_absolute_error(df['widget_sales'].iloc[450:],
                                    df['pred_widgets_sales'].iloc[450:])
print(mae_ma_undiff)
