#!/usr/bin/env python
# coding: utf-8

# 建立字频表：先tokenize 再 freq/rank
# * df_v为 BOW 全局的字频表
# * 注意如何使用pandas模块建立 counts (频率) 及 rank (排名)
# -----------------
# * np.concatenate() 为拼接(axis为从甚么维度上拼接)
# * 此处仅为拼接 lcontent_columns

import numpy as np
import pandas as pd
def tokenize_from_BOW(BOW):
    words_in_a_cropus = list(np.concatenate(list(BOW),axis=0))
    d = pd.DataFrame({'words': words_in_a_cropus}, index = list(range(len(words_in_a_cropus))))
    return(d)

def v_freq_rank(df):
    d = df.groupby(by="words").size().reset_index(name='freq').sort_values(by='freq', ascending=False)
    d['rank'] = d['freq'].rank(method='dense', ascending=False)
    d.reset_index(inplace=True)
    return(d)