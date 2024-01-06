"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-19 16:20:37
@Description: 
"""

import numpy as np
from numpy import ndarray
import pandas as pd


def mape(y_true, y_pred) -> float:
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))


def simulate_process(is_stationary: bool, n: int = 400) -> ndarray:
    np.random.seed(42)

    process = np.zeros(n)
    if is_stationary:
        alpha = .5
    else:
        alpha = 1
    for i in range(n-1):
        process[i+1] = alpha * process[i] + np.random.standard_normal()

    return process


def mean_over_time(ts: ndarray) -> list:
    return [np.mean(ts[:i]) for i in range(1, len(ts))]


def var_over_time(ts: ndarray) -> list:
    return [np.var(ts[:i]) for i in range(1, len(ts))]


def _add_original_feature(df, df_new):
    # df_new['open'] = df['Open']
    df_new['open_1'] = df['Open'].shift(1)
    df_new['close_1'] = df['Close'].shift(1)
    df_new['high_1'] = df['High'].shift(1)
    df_new['low_1'] = df['Low'].shift(1)
    df_new['volume_1'] = df['Volume'].shift(1)


def _add_avg_price(df, df_new):
    df_new['avg_price_5'] = df['Close'].rolling(5).mean().shift(1)
    df_new['avg_price_30'] = df['Close'].rolling(21).mean().shift(1)
    df_new['avg_price_365'] = df['Close'].rolling(252).mean().shift(1)
    df_new['ratio_avg_price_5_30'] = df_new['avg_price_5'] / \
        df_new['avg_price_30']
    df_new['ratio_avg_price_30_365'] = df_new['avg_price_30'] / \
        df_new['avg_price_365']


def _add_avg_volume(df, df_new):
    df_new['avg_volume_5'] = df['Volume'].rolling(5).mean().shift(1)
    df_new['avg_volume_30'] = df['Volume'].rolling(21).mean().shift(1)
    df_new['avg_volume_365'] = df['Volume'].rolling(252).mean().shift(1)
    df_new['ratio_avg_volume_5_30'] = df_new['avg_volume_5'] / \
        df_new['avg_volume_30']
    df_new['ratio_avg_volume_5_365'] = df_new['avg_volume_5'] / \
        df_new['avg_volume_365']
    df_new['ratio_avg_volume_30_365'] = df_new['avg_volume_30'] / \
        df_new['avg_volume_365']


def _add_std_price(df, df_new):
    df_new['std_price_5'] = df['Close'].rolling(5).std().shift(1)
    df_new['std_price_30'] = df['Close'].rolling(30).std().shift(1)
    df_new['std_price_365'] = df['Close'].rolling(252).std().shift(1)
    df_new['ratio_std_price_5_30'] = df_new['std_price_5'] / \
        df_new['std_price_30']
    df_new['ratio_std_price_5_365'] = df_new['std_price_5'] / \
        df_new['std_price_365']
    df_new['ratio_std_price_30_365'] = df_new['std_price_30'] / \
        df_new['std_price_365']


def _add_std_volume(df, df_new):
    df_new['std_volume_5'] = df['Volume'].rolling(5).std().shift(1)
    df_new['std_volume_30'] = df['Volume'].rolling(30).std().shift(1)
    df_new['std_volume_365'] = df['Volume'].rolling(252).std().shift(1)
    df_new['ratio_std_volume_5_30'] = df_new['std_volume_5'] / \
        df_new['std_volume_30']
    df_new['ratio_std_volume_5_365'] = df_new['std_volume_5'] / \
        df_new['std_volume_365']
    df_new['ratio_std_volume_30_365'] = df_new['std_volume_30'] / \
        df_new['std_volume_365']


def _add_return_feature(df, df_new):
    df_new['return_1'] = (df['Close'].apply(np.log).diff(1)).shift(1)
    df_new['return_5'] = (df['Close'].apply(np.log).diff(5)).shift(1)
    df_new['return_30'] = (df['Close'].apply(np.log).diff(21)).shift(1)
    df_new['return_365'] = (df['Close'].apply(np.log).diff(252)).shift(1)
    df_new['moving_avg_5'] = df_new['return_1'].rolling(5).mean()
    df_new['moving_avg_30'] = df_new['return_1'].rolling(21).mean()
    df_new['moving_avg_365'] = df_new['return_1'].rolling(252).mean()


def generate_features(df):
    df_new = pd.DataFrame()
    _add_original_feature(df, df_new)
    _add_avg_price(df, df_new)
    _add_avg_volume(df, df_new)
    _add_std_price(df, df_new)
    _add_std_volume(df, df_new)
    _add_return_feature(df, df_new)

    df_new['close'] = df['Close']
    df_new = df_new.dropna(axis=0)
    return df_new


if __name__ == '__main__':
    SSE = pd.read_csv('../../data/sse.csv', index_col='Date')

    data = generate_features(SSE)
    start_train = '2009-01-01'
    end_train = '2022-12-31'
    start_test = '2023-01-01'
    end_test = '2023-12-31'
    data_train = data.loc[start_train:end_train]
    X_train = data_train.drop('close', axis=1).values
    y_train = data_train['close'].values
    data_test = data.loc[start_test:]
    X_test = data_test.drop('close', axis=1).values
    y_test = data_test['close'].values
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    scaler = StandardScaler()

    X_scales_train = scaler.fit_transform(X_train)
    X_scales_test = scaler.transform(X_test)

    param_grid = {
        'alpha': [1e-4, 3e-4, 1e-3],
        'eta0': [.01, .03, .1],
    }

    lr = SGDRegressor(penalty='l2', max_iter=1_000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_scales_train, y_train)

    # print(grid_search.best_params_)
    lr_best = grid_search.best_estimator_
    predictions_lr = lr_best.predict(X_scales_test)

    print(f'MSE: {mean_squared_error(y_test, predictions_lr):.3f}')

    print(f'MAE: {mean_absolute_error(y_test, predictions_lr):.3f}')
    print(f'R^2: {r2_score(y_test, predictions_lr):.3f}')

    import datetime
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(data_test.index, y_test, c='k', label='Truth', linewidth=1)
    ax.plot(data_test.index, predictions_lr, c='b',
            label='Linear regression', linewidth=1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Close price')
    ax.legend()
    fig.autofmt_xdate()
    plt.xlim([datetime.date(2023, 11, 1), datetime.date(2023, 12, 5)])
    plt.tight_layout()
