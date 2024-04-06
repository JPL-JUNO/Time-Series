"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-12-03 15:06:37
@Description: 
"""
from typing import Union
import pandas as pd
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX


def optimize_arma(endog: Union[pd.Series, list],
                  order_list: list) -> pd.DataFrame:
    results = []
    for order in tqdm_notebook(order_list):
        try:
            model = SARIMAX(endog, order=(order[0], 0, order[1]),
                            simple_differencing=False).fit(disp=False)
        except:
            continue
        aic = model.aic
        results.append([order, aic])
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, q)', 'AIC']

    result_df = result_df.sort_values(
        by='AIC', ascending=True).reset_index(drop=True)

    return result_df
