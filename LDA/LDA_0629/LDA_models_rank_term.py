#!/usr/bin/env python
# coding: utf-8

import pandas as pd

def rank_term_context(BOW, term, pre_post = 'pre'):
    """
    看看甚么样的XX研究
    可以是XX+，也可以是+XX
    """
    if pre_post == 'pre':
        research_kinds = [c[c.index(term)-1] for c in BOW if term in c] 
    else:
        research_kinds = [c[c.index(term)+1] for c in BOW if term in c if c.index(term)+1<len(c)] 
        
    df_rk =  pd.DataFrame(list(zip(research_kinds,[1]*len(research_kinds))))
    df_rk.columns = [pre_post,term]
    out = df_rk.groupby(by=pre_post).count().sort_values(by=term, ascending=False)
    return (out)


if __name__ == '__main__':
    rank_term_context()

