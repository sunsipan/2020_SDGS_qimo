#!/usr/bin/env python
# coding: utf-8

def replace_term(BOW, term, term_new):
    out_list_all = []
    for x in BOW:
        out_list = []
        for e in x:
            if e==term:
                out_list.append(term_new)
            else:
                out_list.append(e) 
        out_list_all.append(out_list)
    return (out_list_all)

if __name__ == '__main__':
    replace_term()

