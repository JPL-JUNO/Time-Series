"""
@Title: 随机游走
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-21 21:30:55
@Description: 
"""

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

steps = np.random.standard_normal(1_000)
steps[0] = 0
random_work = steps.cumsum()

fig, ax = plt.subplots()
ax.plot(random_work)
ax.set_xlabel('Timesteps')
ax.set_ylabel('Value')
fig.tight_layout()
