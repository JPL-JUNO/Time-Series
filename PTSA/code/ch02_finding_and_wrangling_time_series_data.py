import pandas as pd

emails = pd.read_csv("../data/emails.csv", parse_dates=["week"])
complete_idx = pd.MultiIndex.from_product((set(emails.week), set(emails.user)))

all_email = (
    emails.set_index(["week", "user"])
    .reindex(complete_idx, labels=["week", "user"], fill_value=0)
    .reset_index()
)
