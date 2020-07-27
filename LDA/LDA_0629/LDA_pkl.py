import pickle
import pandas as pd
fn = { "LDA_pickle_n": "pkl/LDA_{n}.pickle" }

def load_model_from_pkl(filename, filtered=True):
    with open(filename, 'rb') as fp:            
        model_data = pickle.load(fp)
    if filtered==True:
        # pyLDAvis可传的参数仅限以下列表，所以需过滤
        model_data['vocab'] = model_data.pop('term')     # pyLDAvis model要求叫做vocav，不是 term
        params_allowed = ['doc_lengths', 'vocab', 'term_frequency', 'topic_term_dists', 'doc_topic_dists']
        model_data = { k:v for k,v in model_data.items() if k in params_allowed }
    else:
        pass
    return(model_data)

def load_doc_topic_dists (N, topic_label="主题", sort_columns=True):
    fin_tmpl = fn ["LDA_pickle_n"]
    filename = fin_tmpl.format(n=N)

    model_data = load_model_from_pkl(filename, filtered=False) # False to get ['corpus_index']

    dtM = pd.DataFrame(model_data["doc_topic_dists"], index= model_data['corpus_index'])
    
    # 和pyLDA的结果匹配，從最重要的主题開始並從1開始
    if sort_columns==True:
        columns_new = list(dtM.sum(axis=0).sort_values(ascending=False).index)
        dtM = dtM[columns_new]
        start = 1 # 從1開始
        dtM.columns = list(range(start,len(dtM.columns)+start))  
    # -------------------------------------------------------------
        
    return(dtM)


if __name__ == '__main__':
    print ("module LDA_pkl")