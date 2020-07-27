#!/usr/bin/env python
# coding: utf-8

import pandas as pd

d = dict()

def read_freq(xls_fn):
    """
    读取xls_freq的所有sheet表格文件
    """
    with pd.ExcelFile(xls_fn) as xls:
        all_sheets = xls.sheet_names
    print (all_sheets)
    for sheet in all_sheets:
        d[sheet] = pd.read_excel(xls_fn, sheet_name=sheet, index_col=0)
    return d


def eval_index(dataset_df,index_list):
    for c in index_list:
        dataset_df[c] = dataset_df[c].apply(lambda x: eval(x))


if __name__ == '__main__':
    read_freq()
    eval_index()
