"""
@Title: Testing for stationarity
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-23 22:10:25
@Description: 
"""

import numpy as np
import matplotlib.pyplot as plt
from utils.funcs import simulate_process
from utils.ts_plot import simulate_process_plot
from utils.funcs import mean_over_time, var_over_time

stationary = simulate_process(is_stationary=True)
non_stationary = simulate_process(is_stationary=False)

fig, ax = plt.subplots()
ax = simulate_process_plot(ax, stationary)
ax = simulate_process_plot(ax, non_stationary, label='non-stationary')
ax.set_ylabel('Value')
plt.savefig('figures/ch03_simulate_stationary.png', dpi=300)


stationary_mean = mean_over_time(stationary)
non_stationary_mean = mean_over_time(non_stationary)

fig, ax = plt.subplots()
ax = simulate_process_plot(ax, stationary_mean)
ax = simulate_process_plot(ax, non_stationary_mean, label='non-stationary')
ax.set_ylabel('Mean')
plt.savefig('figures/ch03_mean_of_stationary.png', dpi=300)

stationary_var = var_over_time(stationary)
non_stationary_var = var_over_time(non_stationary)

fig, ax = plt.subplots()
ax = simulate_process_plot(ax, stationary_var)
ax = simulate_process_plot(ax, non_stationary_var)
ax.set_ylabel('Variance')
plt.tight_layout()
plt.savefig('figures/ch03_variance_of_stationary.png', dpi=300)
