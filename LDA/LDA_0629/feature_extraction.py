#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer     # 文本矢量化

# 文本矢量化 LDA不需用tf-idf，用tf即可
# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
def doc_vectorizer (corpus, max_df, min_df, max_features):
    
    tf_vectorizer = CountVectorizer(max_df=0.5, min_df = min_df, max_features=max_features) #0.085
    
    # doc_term_dists
    
    doc_term_dists = tf_vectorizer.fit_transform(list(corpus.values()))
    LDA_terms = tf_vectorizer.get_feature_names()
    
    _model_= dict()
    _model_['corpus_index'] = list(corpus.keys())
    _model_['doc_lengths'] = [len(x.split(" ")) for x in list(corpus.values())]
    _model_['term'] = LDA_terms
    _model_['term_frequency'] = list(doc_term_dists.toarray().sum(axis=0))
    _model_['doc_term_dists'] = doc_term_dists
    return(_model_)



# 产出带有字词内容的 文本-字词 doc_term 矩阵
# 为进一步产出字词内容"重要性"指标做准备
def gen_df_doc_term_matrix(doc_term_dists, terms, corpus_index=None ): 
    df = pd.DataFrame(doc_term_dists.toarray())
    df.columns = list(terms)
    df.columns.name = "term"          # 好习惯，有表现力的df
    
    if corpus_index is None:
        df.index.name="index"         # 好习惯，有表现力的df
    else:
        df.index = corpus_index
        df.index.name="corpus_index"  # 好习惯，有表现力的df
    return (df)
    
def gen_df_doc_term_matrix_from_model(_model_, kind="index"): # kind="corpus_index" or "index"
    try:
        doc_term_dists = _model_["doc_term_dists"]
        terms = _model_['term']
    except:
        print ("Warning: model missing values for doc_term_dists or term")
    try:
        if kind=="corpus_index":
            try:
                corpus_index = _model_["corpus_index"]
            except:
                print ("Warning: model missing corpus_index")                
            output = gen_df_doc_term_matrix(doc_term_dists, terms, corpus_index=corpus_index)
        else:
            output = gen_df_doc_term_matrix(doc_term_dists, terms)
    except:
        print ("Errors in generating matrix")
        output = None
    
    return(output)   



# 随机森林  文本-字词矩阵：降维

from sklearn.ensemble import RandomForestRegressor

def gen_df_importance_using_RandomForest(df_doc_term_matrix, max_depth=30): # 输入 df_doc_term_matrix
    model = RandomForestRegressor(random_state=0, max_depth=max_depth)  # RandomForest模型生成器(空)，预设参数random_state, max_depth
    y = pd.get_dummies(df_doc_term_matrix)
    RF_Regressor = model.fit(df_doc_term_matrix, y)                      # RF_Regressor生成 (有数据后训练出来的结果)
    features = df_doc_term_matrix.columns                               # 特徵值就是terms
    importances = model.feature_importances_                            # 取出训练出的成果, array格式
    
    # 以下分行步骤: (1)将成果两列表zip打包成字典后成数据框, (2)先设'feature'为索引, (3)按重要性由高至底排序, (4)使用reset_index
    df_out = pd.DataFrame(zip(importances, features), 
                          columns=["importance", "feature"])\
               .set_index('feature')\
               .sort_values(by='importance', ascending=False)\
               .reset_index()
    df_out.index = pd.RangeIndex(start=1, stop=len(df_out)+1, step=1)  # index从1 开始
    df_out.index.name = "rank_importance"
    df_out.columns.name = "feature"
    return (df_out)


# prepare for labelling 
def gen_df_term_coder(dfin, label="feature", columns_name=None):
    df = dfin.copy()
    df["类别"] = ""  # term_type
    df["修正"] = ""  # replace by  
    df["memo"] = ""
    df['label'] = df[label]
    if columns_name==None:
        pass
    else:
        df.columns.name = columns_name
    return(df)


if __name__ == '__main__':
    print ("module feature extraction")

