"""
@Title        : 
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2023-12-12 21:11:47
@Description  : 
"""

from typing import Union
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas import DataFrame, Series
import pandas as pd


def optimize(endog: Union[Series, list], order_list: list, **kwargs) -> DataFrame:

    results = []

    for order in tqdm_notebook(order_list):
        if 'd' in kwargs:
            orders = (order[0], kwargs['d'], order[1])
        try:
            model = SARIMAX(endog, order=orders,
                            simple_differencing=False).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p,q)', 'AIC']

    result_df = result_df.sort_values(
        by='AIC', ascending=True).reset_index(drop=True)

    return result_df


if __name__ == '__main__':
    pass
