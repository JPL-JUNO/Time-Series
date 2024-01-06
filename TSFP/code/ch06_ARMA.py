"""
@Title: Modeling complex time series
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-12-01 22:37:49
@Description: 
"""

from itertools import product
from utils.arma import optimize_arma
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np

np.random.seed(42)

ar1 = np.array([1, -.33])
ma1 = np.array([1, .9])

ARMA_1_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=1_000)

ADF_result = adfuller(ARMA_1_1)
print(f'ADF statistics: {ADF_result[0]:.3f}')
print(f'p-value: {ADF_result[1]:.3f}')

fig = plot_acf(ARMA_1_1, lags=20)
fig = plot_pacf(ARMA_1_1, lags=20)


ps = range(0, 4, 1)
qs = range(0, 4, 1)

order_list = list(product(ps, qs))

result_df = optimize_arma(ARMA_1_1, order_list)

# 实现残差分析
model = SARIMAX(endog=ARMA_1_1, order=(1, 0, 1), simple_differencing=False)
model_fit = model.fit(disp=False)
residuals = model_fit.resid

fig = qqplot(residuals, line='45')
model_fit.plot_diagnostics(figsize=(10, 8))

lb_test = acorr_ljungbox(residuals, np.arange(1, 11, 1))
print(lb_test)
