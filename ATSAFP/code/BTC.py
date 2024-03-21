import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bitcoin = pd.read_excel("../data/BitcoinPrice17-6-23-18-6-22.xlsx", header=0)
date = pd.date_range("2017-06-23", periods=len(bitcoin), freq="D")
bitcoin.index = date
price = bitcoin["ClosingP"]
price.plot()
plt.title("Bitcoin Price 2017.6.23 2018.6.22")
plt.ylabel("Price in USD")
plt.savefig(
    "../img/pyTSA_BTC_Fig1-1.png", dpi=1200, bbox_inches="tight", transparent=True
)
