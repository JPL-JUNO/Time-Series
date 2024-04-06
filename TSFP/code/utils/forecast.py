from typing import Union
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


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
