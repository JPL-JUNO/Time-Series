import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

milk = pd.read_excel("../data/milk.xlsx", header=None)
# 将 DataFrame 转为一列
# 绝对不应该这样写
# mseries = pd.concat(
#     [
#         milk.loc[0],
#         milk.loc[1],
#         milk.loc[2],
#         milk.loc[3],
#         milk.loc[4],
#         milk.loc[5],
#         milk.loc[6],
#         milk.loc[7],
#         milk.loc[8],
#         milk.loc[9],
#         milk.loc[10],
#         milk.loc[11],
#         milk.loc[12],
#         milk.loc[13],
#         milk.loc[14],
#         milk.loc[15],
#         milk.loc[16],
#     ],
#     ignore_index="true",
# )
mseries = pd.Series(milk.values.reshape(1, -1)[0])
mts = mseries.dropna()
timeindex = pd.date_range("1962-01", periods=mts.shape[0], freq="M")
mts.index = timeindex

mts.plot()
plt.title("Milk Average per Cow from Jan.1962 to Dec.1975")
plt.xlabel("Time")
plt.ylabel("Milk Average")
plt.savefig(
    "../img/pyTSA_MilkAv_Fig1-2.png",
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)

my = np.array(mts).reshape(-1, 12)
myt = np.transpose(my)
year = np.arange(1962, 1976, 1)
myt = pd.DataFrame(myt, columns=year)
