"""
@Title: 修改文件名
@Author(s): Stephen CUI
@LastEditor(s): Stephen CUI
@CreatedTime: 2023-11-19 13:42:40
@Description: 小写，并用下划线替换空格
"""

import os
for file in os.listdir():
    os.rename(file, file.lower().replace(' ', '_'))
