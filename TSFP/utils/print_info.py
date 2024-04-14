"""
@File         : print_info.py
@Author(s)    : Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime  : 2024-04-14 22:19:55
@Email        : cuixuanstephen@gmail.com
@Description  : 
"""


def print_adf_result(res, variable):
    print(variable)
    print(f"ADF Statistic: {res[0]}")
    print(f"p-value: {res[1]}")
