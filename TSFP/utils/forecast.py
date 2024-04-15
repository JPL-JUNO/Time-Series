from typing import Union
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX


def rolling_forecast(
    endog: Union[pd.Series, list],
    exog: Union[pd.Series, list],
    train_len: int,
    horizon: int,
    window: int,
    method: str,
) -> list:
    total_len = train_len + horizon
    if method == "last":
        pred_last_value = []
        for i in range(train_len, total_len, window):
            last_value = endog[:i].iloc[-1]
            pred_last_value.extend(last_value for _ in range(window))
        return pred_last_value
    elif method == "SARIMAX":
        pred_SARIMAX = []
        for i in range(train_len, total_len, window):
            model = SARIMAX(
                endog[:i],
                exog[:i],
                order=(3, 1, 3),
                seasonal_order=(0, 0, 0, 4),
                simple_differencing=False,
            )
            res = model.fit(disp=False)
            predictions = res.get_prediction(exog=exog)
            oss_pred = predictions.predicted_mean.iloc[-window:]
            pred_SARIMAX.extend(oss_pred)
        return pred_SARIMAX


def rolling_forecast_varmax(
    df: pd.DataFrame, train_len, horizon: int, window: int, method: str, order: tuple
) -> list:
    total_len = train_len + horizon
    series1 = []
    series2 = []
    for i in range(train_len, total_len, window):
        if method == "VAR":
            model = VARMAX(df[:i], order=order)
            res = model.fit(disp=False)
            predictions = res.get_prediction(0, i + window - 1)

            oos_pred_ser1 = predictions.predicted_mean.iloc[-window:]["realdpi"]
            oos_pred_ser2 = predictions.predicted_mean.iloc[-window:]["realcons"]

            series1.extend(oos_pred_ser1)
            series2.extend(oos_pred_ser2)
        elif method == "last":
            # For the baseline, we’ll also use two lists
            # to hold the predictions for each variable and return them at the end.
            ser1_last = df[:i].iloc[-1]["realdpi"]
            ser2_last = df[:i].iloc[-1]["realcons"]
            series1.extend(ser1_last for _ in range(window))
            series2.extend(ser2_last for _ in range(window))
    return series1, series2
