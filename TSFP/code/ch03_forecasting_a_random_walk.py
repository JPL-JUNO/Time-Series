"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-26 20:16:07
@Description: 
"""

from sklearn.metrics import mean_squared_error
from utils.ts_plot import plot_pred, plot_metrics_compare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(42)

steps = np.random.standard_normal(1_000)
steps[0] = 0

random_walk = np.cumsum(steps)

df = pd.DataFrame({'value': random_walk})

train = df[:800]
test = df[800:].copy()

fig, ax = plt.subplots()
ax.plot(random_walk)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
ax.axvspan(800, 1_000, color='#808080', alpha=.2)
plt.tight_layout()

# 用历史均值进行预测
mean = np.mean(train.value)
test.loc[:, 'pred_mean'] = mean

# 用最后一个值进行预测
last_value = train.iloc[-1].value
test.loc[:, 'pred_last'] = last_value

delta_X = len(train) - 1
delta_Y = (train.iloc[-1] - train.iloc[0]).value
drift = delta_Y / delta_X

x_vals = np.arange(800, 1000, 1)
pred_drift = drift * x_vals
test.loc[:, 'pred_drift'] = pred_drift

fig, ax = plot_pred(test)
ax.plot(train.value, 'b-')
ax.plot(test['value'], 'b-')
ax.axvspan(800, 1_000, color='#808080', alpha=.2)
# 这表明随机游走是完全不可以预测的

# 不能使用 MAPE 的原因在于，随机游走可以取到 0
mse_mean = mean_squared_error(test['value'], test['pred_mean'])
mse_last = mean_squared_error(test['value'], test['pred_last'])
mse_drift = mean_squared_error(test['value'], test['pred_drift'])

fig, ax = plot_metrics_compare(x=['mean', 'last', 'drift'],
                               y=[mse_mean, mse_last, mse_drift])


df_shift = df.shift(periods=1)

fig, ax = plt.subplots()
ax.plot(df, 'b-', label='actual')
ax.plot(df_shift, 'r-.', label='forecast')
ax.legend()
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
plt.tight_layout()

mse_one_step = mean_squared_error(test['value'], df_shift[800:])


ax.set_xlim(900, 1_000)
ax.set_ylim(15, 28)
