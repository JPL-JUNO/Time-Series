"""
@File         : smoothing_data.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-03-24 22:38:36
@Email        : cuixuanstephen@gmail.com
@Description  : 平滑数据
"""

import pandas as pd

air = pd.read_csv("../data/AirPassengers.csv", parse_dates=True, header=None)
air.columns = ["Date", "Passengers"]

air["Smooth.5"] = air["Passengers"].ewm(alpha=0.5).mean()
air["Smooth.1"] = air["Passengers"].ewm(alpha=0.1).mean()
air["Smooth.9"] = air["Passengers"].ewm(alpha=0.9).mean()
