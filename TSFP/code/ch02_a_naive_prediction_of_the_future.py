"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-19 13:46:58
@Description: 
"""

import pandas as pd
import numpy as np
from utils.funcs import mape
from utils.ts_plot import ts_plot
import matplotlib.pyplot as plt

df = pd.read_csv('../data/jj.csv')

# 列表的 slice 是进行了复制，但是 pd.DataFrame 没有进行复制
train = df[:-4].copy()
test = df[-4:].copy()

historical_mean = np.mean(train['data'])

test.loc[:, ['pred_mean']] = historical_mean

mape_hist_mean = mape(test['data'], test['pred_mean'])

ts_plot(train, test, 'pred_mean', 8, [1960, 1962, 1964, 1966, 1968, 1970, 1972,
                                      1974, 1976, 1978, 1980])

last_year_mean = np.mean(train['data'][-4:])
test.loc[:, ['pred_last_yr_mean']] = last_year_mean

mape_last_year_mean = mape(test['data'], test['pred_last_yr_mean'])

ts_plot(train, test, 'pred_last_yr_mean', 8,
        [1960, 1962, 1964, 1966, 1968, 1970, 1972,
         1974, 1976, 1978, 1980])


last = train['data'].iloc[-1]
test.loc[:, ['pred_last']] = last
mape_last = mape(test['data'], test['pred_last'])
ts_plot(train, test, 'pred_last', 8,
        [1960, 1962, 1964, 1966, 1968, 1970, 1972,
         1974, 1976, 1978, 1980])


test.loc[:, ['pred_last_season']] = train['data'].iloc[-4:].values
mape_naive_seasonal = mape(test['data'], test['pred_last_season'])
ts_plot(train, test, 'pred_last_season', 8,
        [1960, 1962, 1964, 1966, 1968, 1970, 1972,
         1974, 1976, 1978, 1980])

fig, ax = plt.subplots()
x = ['hist_mean', 'last_year_mean', 'last', 'naive_seasonal']
y = [70.00, 15.60, 30.46, 11.56]
ax.bar(x, y, width=.4)
ax.set_label('Baselines')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 100)
for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(value),
             ha='center')
plt.tight_layout()
plt.savefig('figures/ch02_baseline.png', dpi=300)
