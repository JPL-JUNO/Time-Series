"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-26 14:42:17
@Description: 
"""

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from utils.ts_plot import simulate_process_plot
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

steps = np.random.standard_normal(400)
steps[0] = 0

random_walk = np.cumsum(steps)

ADF_result = adfuller(random_walk)
# stats, p_value, *others = adfuller(random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(random_walk, lags=20)

diff_random_walk = np.diff(random_walk, n=1)

fig, ax = plt.subplots()
ax = simulate_process_plot(ax, diff_random_walk)
ax.set_ylabel('Value')

ADF_result = adfuller(diff_random_walk)
print(f'ADF Statistic: {ADF_result[0]}')
print(f'p-value: {ADF_result[1]}')

plot_acf(diff_random_walk, lags=20)
