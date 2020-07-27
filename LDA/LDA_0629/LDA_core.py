#!/usr/bin/env python
# coding: utf-8

import pandas as pd

from sklearn.decomposition import NMF, LatentDirichletAllocation  # LDA主题模型

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


# --------------------------------------------------------------------------------------
# LDA主题模型 指定要几个主题，预设值N=7
def LDA_run(_model_, N=7):
    # Run LDA
    doc_term_dists = _model_['doc_term_dists']
    
    lda = LatentDirichletAllocation(n_components=N, max_iter=500, \
                                    learning_method='online', \
                                    learning_offset=50., \
                                    random_state=0).fit(doc_term_dists)
    
    _model_['topic_term_dists']  = lda.components_  # .shape
    _model_['doc_topic_dists']  = lda.transform(doc_term_dists) 
    
    return(_model_, lda)

# LDA主题模型加工
# 产出Topics主题的表格
def gen_df_topic(LDA, feature_names, no_top_words=30, splitter="丶"):
    d = { topic_idx: splitter.join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]) \
          for topic_idx, topic in enumerate(LDA.components_)   }
    topic_term = pd.DataFrame({"terms":d})
    topic_term.index.name  = "topic_index"   
    topic_term.columns.name = "topic_top_{n}".format(n=no_top_words)
    return (topic_term)

# 产出最简化的文本-主题-赋值矩阵
def gen_df_doc_topic(_model_, LDA):  
    doc_term_dists = _model_['doc_term_dists']
    doc_topic = pd.DataFrame(LDA.transform(doc_term_dists))
    doc_topic.index = _model_['corpus_index']
    doc_topic.index.name="doc"
    doc_topic.columns.name="topic"    
    return(doc_topic)    

# 执行以上, 产出所有
import feature_extraction as FE

def LDA_run_all(corpus, N, max_df=0.5, min_df=0.1, max_features=50000):  
    # 特徵值门槛
    M = FE.doc_vectorizer (corpus, max_df, min_df, max_features )
    no_topics = N
    M, LDA = LDA_run (M, N=no_topics)
    topic_term = gen_df_topic(LDA, M['term'], no_top_words=30)
    doc_topic  = gen_df_doc_topic(M, LDA)
    return (M, LDA,topic_term, doc_topic)    



if __name__ == '__main__':
    print ("LDA_core.py")

