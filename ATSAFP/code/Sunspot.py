import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv(
    "../data/Yearly mean total sunspot number 1700 - 2017.csv",
    delimiter=";",
    header=None,
    index_col=0,
)
sunspot = x[1]
sunspot.plot(legend=False, color="green")
plt.title("Yearly mean total sunspot number")
plt.ylabel("Sunspot number")
plt.xlabel("Year")
plt.savefig(
    "../img/Sunspot_number.png",
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)
plt.show()
