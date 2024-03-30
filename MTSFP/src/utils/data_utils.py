import pandas as pd
from collections import defaultdict
from tqdm.autonotebook import tqdm
import warnings
import numpy as np


def compact_to_expanded(
    df, timeseries_col, static_cols, time_varying_cols, ts_identifier
):
    def preprocess_expanded(x):
        dr = pd.date_range(
            start=x["start_timestamp"],
            periods=len(x["energy_consumption"]),
            freq=x["frequency"],
        )
        df_columns = defaultdict(list)
        df_columns["timestamp"] = dr
        for col in [ts_identifier, timeseries_col] + static_cols + time_varying_cols:
            df_columns[col] = x[col]
        return pd.DataFrame(df_columns)

    all_series = []
    for i in tqdm(range(len(df))):
        all_series.append(preprocess_expanded(df.iloc[i]))
    df = pd.concat(all_series)
    del all_series
    return df


def write_compact_to_ts(
    df,
    filename,
    static_columns,
    time_varying_columns,
    sep: str = ",",
    encoding: str = "utf-8",
    date_format: str = "%Y-%m-%d %H-%M-%S",
    chunk_size: int = 50,
):
    if sep == ":":
        warnings.warn(
            "Using `:` as separator will not work well if `:` is present in the string representation of date time."
        )
    with open(filename, "w", encoding=encoding) as f:
        for c, dtype in df.dtypes.items():
            if c in static_columns:
                typ = "static"
            elif c in time_varying_columns:
                typ = "time_varying"
            f.write(f"@column {c} {dtype.name} {typ}")
            f.write("\n")
        f.write(f"@data")
        f.write("\n")

    def write_ts(x):
        l = ""
        for c in x.index:
            if isinstance(x[c], np.ndarray):
                l += "|".join(x[c].astype(int))
                l += "sep"
            elif isinstance(x[c], pd.Timestamp):
                l += x[c].strftime(date_format)
                l += sep
            else:
                l += str(x[c])
                l += sep
        l += "\n"
        return l

    [
        f.writelines([write_ts(x.loc[i])] for i in tqdm(x.index))
        for x in tqdm(
            np.split(df, np.arange(chunk_size, len(df), chunk_size)),
            desc="Writing in Chunks...",
        )
    ]
