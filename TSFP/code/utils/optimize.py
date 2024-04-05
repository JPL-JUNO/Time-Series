from typing import Union
from tqdm import tqdm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd


def optimize_SARIMA(
    endog: Union[pd.Series, list], order_list: list, d: int, D: int, s: int
) -> pd.DataFrame:
    results = []
    for order in tqdm(order_list):
        arima_order = (order[0], d, order[1])
        if len(order) == 4:
            seasonal_order = (order[2], D, order[3], s)
        elif len(order) == 2:
            seasonal_order = (0, D, 0)
        try:
            model = SARIMAX(
                endog,
                order=arima_order,
                seasonal_order=seasonal_order,
                simple_differencing=False,
            ).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q,P,Q)", "AIC"]

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df


def optimize_SARIMAX(
    endog: Union[pd.Series, list],
    exog: Union[pd.Series, list],
    order_list: list,
    d: int,
    D: int,
    s: int,
) -> pd.DataFrame:
    results = []
    for order in tqdm(order_list):
        arima_order = (order[0], d, order[1])
        if len(order) == 4:
            seasonal_order = (order[2], D, order[3], s)
        elif len(order) == 2:
            seasonal_order = (0, D, 0)

        try:
            model = SARIMAX(
                endog,
                exog,  # Notice the addition of the exogenous variables when fitting the model.
                order=arima_order,
                seasonal_order=seasonal_order,
                simple_differencing=False,
            ).fit(disp=False)
        except:
            continue

        aic = model.aic
        results.append([order, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q,P,Q)", "AIC"]

    # Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)

    return result_df
