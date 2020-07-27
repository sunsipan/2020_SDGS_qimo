#!/usr/bin/env python
# coding: utf-8

import os
from openpyxl import load_workbook
from datetime import datetime
import pandas as pd

def get_timestamped(_str_):
    time8d = datetime.now().strftime('%m%d%H%M')
    return(_str_ + "_" + time8d)

def writer_do_sheets(writer, dict_df, get_time=True):
    for sheet_name, s in dict_df.items():
        if get_time==True:
            s.to_excel(writer, sheet_name=get_timestamped(sheet_name))
        else:
            s.to_excel(writer, sheet_name=sheet_name)

def check_and_write_xls(path, dict_df_xls, get_time=True):
    if os.path.exists(path):    
        with pd.ExcelWriter(path, engine='openpyxl', mode="a") as writer: # 增加
             writer_do_sheets(writer, dict_df_xls, get_time=get_time)
    else:
        with pd.ExcelWriter(path, engine='openpyxl', mode="w") as writer: # 创建
             writer_do_sheets(writer, dict_df_xls, get_time=get_time)
