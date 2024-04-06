"""
@Title: 
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-19 21:29:24
@Description: 
"""
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes


def ts_plot(train, test,
            pred_label: str = 'pred',
            xticks_span: int = 5,
            xticks: list = None) -> Figure:
    train_length = len(train)
    test_length = len(test)
    total_sample = train_length + test_length
    if xticks is None:
        xticks = range(total_sample / xticks_span) + 1
    fig, ax = plt.subplots()
    for (data, shape, label) in zip([train, test], ['g-', 'b-'],
                                    ['train', 'test']):
        ax.plot(data['date'], data['data'], shape, label=label)
    ax.plot(test['date'], test[pred_label], 'r--', label='Predicted')
    ax.set_xlabel('Date')
    ax.set_ylabel('True Value vs Predicted Value')
    ax.axvspan(train_length, train_length + test_length -
               1, color='#808080', alpha=.2)
    ax.legend()
    fig.autofmt_xdate()
    plt.xticks(np.arange(0, total_sample, xticks_span), xticks)
    plt.tight_layout()
    return fig


def simulate_process_plot(ax: Axes, ts: ndarray,
                          label: str = 'stationary') -> Axes:
    ax.plot(ts, label=label)
    ax.set_xlabel('Timesteps')
    ax.legend()
    return ax


def plot_pred(df: ndarray, ways: list = ['mean', 'last', 'drift'],
              shapes: list = ['r-.', 'g--', 'k:']) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    for way, shape in zip(ways, shapes):
        ax.plot(df['pred_' + way], shape, label=way)
    ax.legend()
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Value')
    plt.tight_layout()
    return fig, ax


def plot_metrics_compare(x: list, y: list, metrics: str = 'MSE'):
    fig, ax = plt.subplots()
    ax.bar(x, y, width=.4)
    ax.set_xlabel('Methods')
    ax.set_ylabel(metrics)
    for index, value in enumerate(y):
        plt.text(x=index, y=value*1.03, s=str(round(value, 2)), ha='center')
    plt.tight_layout()
    return fig, ax
