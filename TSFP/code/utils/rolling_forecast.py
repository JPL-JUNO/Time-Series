"""
@Title: 对 MA 模型进行滚动预测
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-30 00:09:05
@Description: 
"""

from pandas import DataFrame
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np


def _naive_mean(df, train_len, total_len, window, order=None):
    pred = []
    for i in range(train_len, total_len, window):
        mean = np.mean(df[:i].values)
        pred.extend(mean for _ in range(window))
    return pred


def _naive_last(df, train_len, total_len, window, order=None):
    pred = []
    for i in range(train_len, total_len, window):
        last_value = df[:i].iloc[-1].values[0]
        pred.extend(last_value for _ in range(window))
    return pred


def _ma(df, train_len, total_len, window, **kwargs):
    order = kwargs.get("order", (3, 0, 0))
    pred = []
    for i in range(train_len, total_len, window):
        model = SARIMAX(df[:i], order=order)
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oos_pred = predictions.predicted_mean.iloc[-window:]
        pred.extend(oos_pred)
    return pred


def _ar(df, train_len, total_len, window, **kwargs):
    order = kwargs.get("order", (3, 0, 0))
    pred = []
    for i in range(train_len, total_len, window):
        model = SARIMAX(df[:i], order=order)
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oos_pred = predictions.predicted_mean.iloc[-window:]
        pred.extend(oos_pred)
    return pred


def _arma(df, train_len, total_len, window, **kwargs):
    order = kwargs.get("order", (2, 0, 2))
    pred = []
    for i in range(train_len, total_len, window):
        model = SARIMAX(df[:i], order=order)
        res = model.fit(disp=False)
        predictions = res.get_prediction(0, i + window - 1)
        oos_pred = predictions.predicted_mean.iloc[-window:]
        pred.extend(oos_pred)
    return pred


def _arima(endog, order_list: list, **kwargs):
    pass


methods = {
    "mean": _naive_mean,
    "last": _naive_last,
    "MA": _ma,
    "AR": _ar,
    "arma": _arma,
    "arima": _arima,
}


def rolling_forecast(
    df: DataFrame, train_len: int, horizon: int, window: int, method: str, **kwargs
) -> list:
    # BUG, 如果预测长度不能被 window 整除，是不是出现没有预测的情况
    total_len = train_len + horizon
    assert method in methods, "预测方法不存在"
    pred = methods.get(method)(df, train_len, total_len, window, **kwargs)
    return pred


if __name__ == "__main__":
    # rolling_forecast()
    pass
