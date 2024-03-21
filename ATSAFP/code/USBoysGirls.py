import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

birth = pd.read_csv("../data/Noboyngirl.csv", index_col=0)
birth_year = birth.set_index("year")

birth_year.plot()
(boy,) = plt.plot(birth_year["boys"], "b-", label="boys")
(girl,) = plt.plot(birth_year["girls"], "r--", label="girls")
plt.legend(handles=[boy, girl])
plt.savefig(
    "../img/pyTSA_USBoysGirls_Fig1-21.png",
    dpi=1200,
    bbox_inches="tight",
    transparent=True,
)
