"""
Generate data associated with the LDA networkx (louvian clustering) plot. 
"""
#%%

import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt 
from gensim.models import LdaModel
from nlp_utils import gensim_utils, sklearn_utils, fileio
from nlp_utils.fileio import load_df_semantic, load_df_SEAMs
import sqlite3
from dotenv import load_dotenv
load_dotenv()

lda_models_folder = os.path.join(os.getenv('REPO_DIR'), r'modeling/lda/models')

# ## SEAMS
# model_name = 'ldamod_seams'
# lda_model_loaded = LdaModel.load(os.path.join(lda_models_folder, model_name + '.lda'))
# db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
# con = sqlite3.connect(db_path)
# df = load_df_SEAMs(con).dropna(subset=['OCR_text'])
# df['inCitations'] = ''

## SOC
model_name = 'ldamod_soc'
lda_model_loaded = LdaModel.load(os.path.join(lda_models_folder, model_name + '.lda'))
db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)
df = load_df_semantic(con, lda_model_loaded.idx)
df['inCitations'] = df['inCitations'].apply(",".join)

#
#%%
print('Generating topic probability matrix')
df_topickeywords, df_doc_topic_probs= gensim_utils.gensim_topic_info(lda_model_loaded, lda_model_loaded.data_words, lda_model_loaded.id2word)
df_edgekeywords, df_doc_edge_probs = gensim_utils.gensim_edge_info(lda_model_loaded, lda_model_loaded.data_words, lda_model_loaded.id2word, df_doc_topic_probs.values)
df_doc_topic_probs.index = df.index
df_doc_edge_probs.index = df.index


if not os.path.exists('data'): os.makedirs('data')
df_topickeywords.to_csv(os.path.join('data','topic_keywords.csv'))
df_edgekeywords.to_csv(os.path.join('data','edge_keywords.csv'))

#Generate graph of topic co-occurence
da_sigma = gensim_utils.calc_cov_wrap(df_doc_topic_probs, df_topickeywords.index.values)
# %%

from pipeline_data_prep import pipeline_data_prep
pipeline_data_prep(df, df_topickeywords, df_doc_topic_probs, df_doc_edge_probs, da_sigma)
