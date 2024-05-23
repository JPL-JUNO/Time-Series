"""
@File         : 03_irregular_resampling.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-05-23 14:39:37
@Email        : cuixuanstephen@gmail.com
@Description  : 不规则数据重采样
"""

import numpy as np
import pandas as pd

n_sales = 1_000
start = pd.Timestamp("2023-01-01 09:00")
end = pd.Timestamp("2024-04-01")

n_days = (end - start).days + 1

irregular_series = pd.to_timedelta(np.random.rand(n_sales) * n_days, unit="D") + start

ts_sales = pd.Series(0, index=irregular_series)
tot_sales = ts_sales.resample("D").count()

# 如果是不需要的时间段怎么处理？
# 比如说非交易时间
