"""
@File         : 04_missing_values.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 14:48:46
@Email        : cuixuanstephen@gmail.com
@Description  : 缺失值
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv(
    "./assets/data/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

series = data["Incoming Solar"].resample("D").sum()

# sample_with_nan = series.head(365 * 2).copy()
sample_with_nan = series.iloc[: 365 * 2].copy()
size_na = int(0.6 * len(sample_with_nan))
idx = np.random.choice(a=range(len(sample_with_nan)), size=size_na, replace=False)
sample_with_nan[idx] = np.nan


# imputation with mean value
avg_value = sample_with_nan.mean()
imp_mean = sample_with_nan.fillna(avg_value)

# imputation with last known observation
# 前 N 条全是缺失值怎么办？
imp_ffill = sample_with_nan.ffill()

# imputation with next known observation
imp_bfill = sample_with_nan.bfill()

plt.rcParams["figure.figsize"] = [12, 8]
fig, (ax0, ax1, ax2, ax3) = plt.subplots(4, sharex=True)

ax0.plot(sample_with_nan)
ax0.set_title("Original series with missing data")
ax1.plot(imp_mean)
ax1.set_title("Series with mean imputation")
ax2.plot(imp_ffill)
ax2.set_title("Series with ffill imputation")
ax3.plot(imp_bfill)
ax3.set_title("Series with bfill imputation")

plt.tight_layout()
plt.savefig("assets/missing_data_plot.png")
