"""
@File         : analyzing_and_visualizing_time_series_data.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-03-29 23:21:15
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""

import numpy as np
import os
from pathlib import Path

os.makedirs("imgs/ch03", exist_ok=True)
preprocessed = Path("data/london_smart_meters/preprocessed")
assert (
    preprocessed.is_dir()
), "You have to run ch02 in Chapter02 before running this notebook"

from src.utils.data_utils import compact_to_expanded
