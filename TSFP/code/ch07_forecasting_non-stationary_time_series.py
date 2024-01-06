"""
@Title        : Forecasting non-stationary time series
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2023-12-12 21:02:31
@Description  : 
"""

from utils.funcs import mape
from utils.ts_plot import plot_metrics_compare, plot_pred
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from utils.ts_predict import optimize
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_csv('../data/jj.csv')
fig, ax = plt.subplots()

ax.plot(df['date'], df['data'])
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
plt.xticks(np.arange(0, 81, 8), [1960, 1962, 1964, 1966, 1968, 1970, 1972,
                                 1974, 1976, 1978, 1980])
fig.autofmt_xdate()
plt.tight_layout()

ad_fuller_result = adfuller(df['data'])
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# 进行差分
eps_diff = np.diff(df['data'], n=1)
ad_fuller_result = adfuller(eps_diff)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

# 再次进行差分
eps_diff2 = np.diff(eps_diff, n=1)
ad_fuller_result = adfuller(eps_diff2)
print(f'ADF Statistic: {ad_fuller_result[0]}')
print(f'p-value: {ad_fuller_result[1]}')

p = range(0, 4, 1)
q = range(0, 4, 1)
d = 2
order_list = list(product(p, q))

train = df['data'][:-4]

result_df = optimize(train, order_list, d=d)

model = SARIMAX(train, order=(3, 2, 3), simple_differencing=False)
model_fit = model.fit(disp=False)

print(model_fit.summary())

fig = model_fit.plot_diagnostics(figsize=(10, 8))

residuals = model_fit.resid

lb_test = acorr_ljungbox(residuals, np.arange(1, 11, 1))

test = df.iloc[-4:].copy()
test['pred_naive_seasonal'] = df['data'].iloc[76:80].values

ARIMA_pred = model_fit.get_prediction(80, 83).predicted_mean

test['pred_arima'] = ARIMA_pred

fig, ax = plot_pred(test, ways=['naive_seasonal', 'arima'])
ax.plot(df['date'], df['data'])
ax.plot(test['data'], 'b-', label='actual')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')
ax.axvspan(80, 83, color='#808080', alpha=0.2)
ax.legend(loc=2)

plt.xticks(np.arange(0, 81, 8), [
           1960, 1962, 1964, 1966, 1968, 1970, 1972, 1974, 1976, 1978, 1980])
ax.set_xlim(60, 83)

fig.autofmt_xdate()
plt.tight_layout()


mape_naive_seasonal = mape(test['data'], test['pred_naive_seasonal'])
mape_arima = mape(test['data'], test['pred_arima'])

fig, ax = plot_metrics_compare(x=['naive_seasonal', 'arima'],
                               y=[mape_naive_seasonal, mape_arima],
                               metrics='MAPE')
ax.set_xlabel('Models')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 15)
plt.tight_layout()
