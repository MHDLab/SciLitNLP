#Want to make a standard set of figures for getting a visualization of parameter choices 

#%%

#%% 
import pandas as pd
import pickle
import sqlite3
import os
import numpy as np
import sys
import gensim
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\nlp_utils')

from nlp_utils.plot import topics_fig
from nlp_utils import gensim_utils
from nlp_utils import sklearn_utils


data_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\data'

con = sqlite3.connect(os.path.join(data_folder, 'nlp.db'))
cursor = con.cursor()

df = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID')

# df = df.sample(500, random_state=1)
df = df.dropna(subset=['processed_text'])

texts = df['processed_text'].values
texts = [t.split() for t in texts]






#%%


params = [
    # {'lda_alpha': 0.3, 'lda_eta':0.1, 'lda_num_topics':20, 'lda_passes': 1, 'bigram_min_count':5, 'bigram_threshold':100},
    [{'alpha': 0.3, 'eta': 0.1, 'num_topics':20, 'passes':1}, {'min_count':5, 'threshold':100}]

]


# lda_kwargs = {key.split('lda_')[1] : param_set[key] for key in param_set.keys() if 'lda_' in key}
# bigram_kwargs = {key.split('bigram_')[1] : param_set[key] for key in param_set.keys() if 'bigram_' in key}


#%%



for i, (lda_kwargs, bigram_kwargs) in enumerate(params): 
    print(lda_kwargs)
    print(bigram_kwargs)


    texts_bigram, id2word, data_words, lda_model = gensim_utils.gensim_lda_bigram(texts, bigram_kwargs, lda_kwargs)

    df_topickeywords, doc_topic_probs = gensim_utils.gensim_topic_info(lda_model, data_words, id2word)

    num_bigrams, total_words = gensim_utils.bigram_stats(data_words, id2word)
    titlestr = str(lda_kwargs) + str(bigram_kwargs) + "\n Num Bigrams: " + str(num_bigrams) + ", Total Words: " + str(total_words) + ", Bigram Fraction: " + str(round(num_bigrams/total_words, 3))

    tsne_x, tsne_y = sklearn_utils.calc_tsne(texts_bigram)

    fig = topics_fig(df_topickeywords, tsne_x, tsne_y, doc_topic_probs, titlestr)

    

    out_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\topic_modeling\compare_params'
    out_path = os.path.join(out_folder, 'fig_' + str(i) + '.png')
    fig.savefig(out_path)


#%%
fig = topics_fig(df_topickeywords, texts_bigram , doc_topic_probs, titlestr)



#%%



#%%


# %%

