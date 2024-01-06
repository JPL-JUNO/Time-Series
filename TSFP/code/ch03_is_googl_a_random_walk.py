"""
@Title: Is GOOGL a random walk?
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-26 18:38:27
@Description: 
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf

df = pd.read_csv('../data/GOOGL.csv')

fig, ax = plt.subplots()
ax.plot(df['Date'], df['Close'])
ax.set_xlabel('Date')
ax.set_ylabel('Closing price (USD)')
plt.xticks(
    [4, 24, 46, 68, 89, 110, 132, 152, 174, 193, 212, 235],
    ['May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov',
     'Dec', 2021, 'Feb', 'Mar', 'April'])
fig.autofmt_xdate()
plt.tight_layout()

GOOGL_ADF_result = adfuller(df['Close'])
print(f'ADF Statistic: {GOOGL_ADF_result[0]:.2f}')
print(f'p-value: {GOOGL_ADF_result[1]:.2f}')

diff_close = np.diff(df['Close'], n=1)
GOOGL_diff_ADF_result = adfuller(diff_close)

print(f'ADF Statistic: {GOOGL_diff_ADF_result[0]:.2f}')
print(f'p-value: {GOOGL_diff_ADF_result[1]:.2f}')

plot_acf(diff_close, lags=20)
