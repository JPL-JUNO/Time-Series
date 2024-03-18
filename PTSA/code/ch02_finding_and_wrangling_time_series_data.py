import pandas as pd

emails = pd.read_csv("../data/emails.csv", parse_dates=["week"])
complete_idx = pd.MultiIndex.from_product((set(emails.week), set(emails.user)))

all_email = (
    emails.set_index(["week", "user"])
    .reindex(complete_idx, fill_value=0)
    .reset_index(names=["week", "user"])
)

all_email[all_email.user == 998].sort_values("week")

cutoff_dates = emails.groupby("user")["week"].agg(["min", "max"]).reset_index()
cutoff_dates

for _, row in cutoff_dates.iterrows():
    user = row["user"]
    start_date = row["min"]
    end_date = row["max"]
    # 这没有问题，因为 bool 类型的过滤会按照 index 匹配，但是不推荐
    # all_email = all_email.drop(
    #     all_email[all_email["user"] == user][all_email["week"] < start_date].index
    # )
    # all_email = all_email.drop(
    #     all_email[all_email["user"] == user][all_email["week"] > end_date].index
    # )
    mask = ((all_email["user"] == user) & (all_email["week"] < start_date)) | (
        (all_email["user"] == user) & (all_email["week"] > end_date)
    )
    # 化简
    # mask = (all_email["user"] == user) & (
    #     (all_email["week"] < start_date) | (all_email["week"] > end_date)
    # )
    all_email = all_email.drop(all_email[mask].index)

donations = pd.read_csv("../data/donations.csv")
donations["timestamp"] = pd.to_datetime(donations["timestamp"])
donations = donations.set_index("timestamp")
agg_don = donations.groupby("user").apply(
    lambda df: df["amount"].resample("W-MON").sum().dropna()
)
agg_don = agg_don.reset_index()

merged_df = []
for user, user_email in all_email.groupby("user"):
    user_donations = agg_don[agg_don["user"] == user]

    user_donations = user_donations.set_index("timestamp")
    user_email = user_email.sort_values("week").set_index("week")

    df = pd.merge(
        user_email, user_donations, how="left", left_index=True, right_index=True
    )
    df = df.fillna(0)
    df["user"] = df["user_x"]
    merged_df.append(df.reset_index())[["user", "week", "emailsOpened", "amount"]]
merged_df = pd.concat(merged_df)
