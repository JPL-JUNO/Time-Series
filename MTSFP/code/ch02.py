from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
import os

tqdm.pandas()

os.makedirs("imgs/ch02", exist_ok=True)
source_data = Path("data/london_smart_meters/")
block_data_path = source_data / "hhblock_dataset" / "hhblock_dataset"
assert block_data_path.is_dir(), "检查数据是否存在"
block_1 = pd.read_csv(block_data_path / "block_0.csv", parse_dates=False)

block_1["day"] = pd.to_datetime(block_1["day"], yearfirst=True)

max_date = None
for f in tqdm(block_data_path.rglob("*.csv")):
    df = pd.read_csv(f, parse_dates=False)
    df["day"] = pd.to_datetime(df["day"], yearfirst=True)
    if max_date is None:
        max_date = df["day"].max()
    else:
        if df["day"].max() > max_date:
            max_date = df["day"].max()
print(f"Max Date across all block: {max_date}")


def preprocess_compact(x):
    # 想要生成一个由开始时间到最大时间长度以及 48 个区段的 index（包含缺失值），这里的处理太麻烦了
    # 直接 from_product() 生成多重索引即可
    start_date = x["day"].min()
    name = x["LCLid"].unique()[0]

    dr = pd.date_range(start=start_date, end=max_date, freq="1D")
    dr = (
        pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr)
        .unstack()
        .reset_index()
    )
    dr.columns = ["hour_block", "day", "_"]
    dr = dr.merge(x, on=["hour_block", "day"], how="left")
    dr = dr.sort_values(["day", "offset"])
    ts = dr["energy_consumption"].values
    len_ts = len(ts)
    return start_date, name, ts, len_ts


def load_process_block_compact(
    block_df, freq="30min", ts_identifier="series_name", value_name="series_value"
):
    grps = block_df.groupby("LCLid")
    all_series = []
    all_start_dates = []
    all_names = []
    all_data = {}
    all_len = []
    for _, df in tqdm(grps, leave=False):
        start_date, name, ts, len_ts = preprocess_compact(df)
        all_series.append(ts)
        all_start_dates.append(start_date)
        all_names.append(name)
        all_len.append(len_ts)

    all_data[ts_identifier] = all_names
    all_data["start_timestamp"] = all_start_dates
    all_data["frequency"] = freq
    all_data[value_name] = all_series
    all_data["series_length"] = all_len
    return pd.DataFrame(all_data)


def preprocessed_expanded(x):
    start_date = x["day"].min()
    dr = pd.date_range(start=start_date, end=x["day"].max(), freq="1D")
    dr = (
        pd.DataFrame(columns=[f"hh_{i}" for i in range(48)], index=dr)
        .unstack()
        .reset_index()
    )
    dr.columns = ["hour_block", "day", "_"]
    dr = dr.merge(x, on=["hour_block", "day"], how="left")
    dr["series_length"] = len(dr)
    return dr


def load_process_block_expanded(block_df, freq="30min"):
    grps = block_df.groupby("LCLid")
    all_series = []
    for _, df in tqdm(grps, leave=False):
        ts = preprocessed_expanded(df)
        all_series.append(ts)
    block_df = pd.concat(all_series)
    block_df["offset"] = block_df["hour_block"].str.replace("hh_", "").astype(int)
    block_df["timestamp"] = block_df["day"] + pd.to_timedelta(
        block_df["offset"] * 30, unit="m"
    )
    block_df["frequency"] = freq
    block_df = block_df.sort_values(["LCLid", "timestamp"])
    block_df = block_df.drop(columns=["_", "hour_block", "offset", "day"])
    return block_df


block_df_1 = []
for file in tqdm(
    sorted(list(block_data_path.glob("*.csv"))), desc="Processing Blocks..."
):
    block_df = pd.read_csv(file, parse_dates=False)
    block_df["day"] = pd.to_datetime(block_df["day"], yearfirst=True)

    block_df = block_df.loc[block_df["day"] >= "2012-01-01"]
    block_df = (
        block_df.set_index(["LCLid", "day"])
        .stack()
        .reset_index()
        .rename(columns={"level_2": "hour_block", 0: "energy_consumption"})
    )
    block_df["offset"] = block_df["hour_block"].str.replace("hh_", "").astype(int)
    block_df_1.append(
        load_process_block_compact(
            block_df,
            freq="30min",
            ts_identifier="LCLid",
            value_name="energy_consumption",
        )
    )
hhblock_df = pd.concat(block_df_1)
del block_df_1

household_info = pd.read_csv(source_data / "informations_households.csv")
hhblock_df = hhblock_df.merge(household_info, on="LCLid", validate="one_to_one")

bank_holidays = pd.read_csv(source_data / "uk_bank_holidays.csv", parse_dates=False)
bank_holidays["Bank holidays"] = pd.to_datetime(
    bank_holidays["Bank holidays"], yearfirst=True
)
bank_holidays = bank_holidays.set_index("Bank holidays")

bank_holidays = bank_holidays.resample("30min").asfreq()
bank_holidays = (
    bank_holidays.groupby(bank_holidays.index.date).ffill().fillna("NO_HOLIDAY")
)
bank_holidays.index.name = "datetime"

weather_hourly = pd.read_csv(
    source_data / "weather_hourly_darksky.csv", parse_dates=False
)
weather_hourly["time"] = pd.to_datetime(weather_hourly["time"], yearfirst=True)
weather_hourly = weather_hourly.set_index("time")

weather_hourly = weather_hourly.resample("30min").ffill()


def map_weather_holidays(row):
    date_range = pd.date_range(
        row["start_timestamp"], periods=row["series_length"], freq=row["frequency"]
    )
    std_df = pd.DataFrame(index=date_range)
    holidays = std_df.join(bank_holidays, how="left").fillna("NO_HOLIDAY")
    weather = std_df.join(weather_hourly, how="left")
    assert len(holidays) == row["series_length"]
    assert len(weather) == row["series_length"]
    row["holidays"] = holidays["Type"].values
    for col in weather:
        row[col] = weather[col].values
    return row


hhblock_df = hhblock_df.progress_apply(map_weather_holidays, axis=1)

del block_df, weather_hourly, bank_holidays, household_info

os.makedirs("data/london_smart_meters/preprocessed", exist_ok=True)
from src.utils.data_utils import write_compact_to_ts

write_compact_to_ts(
    hhblock_df,
    static_columns=[
        "LCLid",
        "start_timestamp",
        "frequency",
        "series_length",
        "stdorToU",
        "Acorn",
        "Acorn_grouped",
        "file",
    ],
    time_varying_columns=[
        "energy_consumption",
        "holidays",
        "visibility",
        "windBearing",
        "temperature",
        "dewPoint",
        "pressure",
        "apparentTemperature",
        "windSpeed",
        "precipType",
        "icon",
        "humidity",
        "summary",
    ],
    filename=f"data/london_smart_meters/preprocessed/london_smart_meters_merged.ts",
    sep=";",
    chunk_size=1,
)
hhblock_df[["LCLid", "file", "Acorn_grouped"]].to_pickle(
    f"data/london_smart_meters/preprocessed/london_smart_meters_lclid_acorn_map.pkl"
)
blocks = [f"block_{i}" for i in range(111)]

n_chunks = 8
split_blocks = [blocks[i : i + n_chunks] for i in range(0, len(blocks), n_chunks)]
for blk in tqdm(split_blocks):
    df = hhblock_df.loc[hhblock_df.file.isin(blk)]
    blk = [int(b.replace("block_", "")) for b in blk]
    block_str = f"block_{min(blk)}-{max(blk)}"
    df.to_parquet(
        f"data/london_smart_meters/preprocessed/london_smart_meters_merged_{block_str}.parquet"
    )
