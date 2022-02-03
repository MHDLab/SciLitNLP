#%%

import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt 
from gensim.models import LdaModel
from nlp_utils.fileio import load_df_semantic
import nlp_utils as nu
import sqlite3
import _pickle as cPickle
from dotenv import load_dotenv
load_dotenv()

db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

#%%


## Corex
models_folder = os.path.join(os.getenv('REPO_DIR'), r'modeling/corex/models')
model_path = os.path.join(models_folder, 'corexmod_soc.pkl')
topic_model = cPickle.load(open(model_path, 'rb'))
s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)


df_tm = load_df_semantic(con, topic_model.docs)
df_tm['inCitations'] = df_tm['inCitations'].apply(",".join)

df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)
# highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]
highlight_topics = []

## LDA

# df_topickeywords, doc_topic_probs = nu.gensim_utils.gensim_topic_info(lda_model_loaded, lda_model_loaded.data_words, lda_model_loaded.id2word)
# df_doc_topic_probs = pd.DataFrame(doc_topic_probs, columns=df_topickeywords.index, )
# df_doc_topic_probs.index = df.index
# df_doc_topic_probs

# years = list(set(df['Year'].dropna()))
# df_topicsyear = pd.DataFrame(index=years, columns=df_topickeywords.index, dtype=float)
# df_topicsyear.index.name = 'year'

# for year in years: 
#     ids = df[df['Year'] == year].index
#     topics_year = df_doc_topic_probs.loc[ids].sum()
#     topics_year = topics_year/topics_year.sum()
#     df_topicsyear.loc[year] = topics_year

# #normailze each topic by it's relative weight (cancels year normalization, but allows for better comparison of slopes)
# #TODO: just normalize slope by sum after fitting? 
# for topic_id in df_topicsyear:
#     df_topicsyear[topic_id] = df_topicsyear[topic_id]/sum(df_topicsyear[topic_id])

# s_topic_words = df_topickeywords.apply(" ".join, axis=1)
# highlight_topics = []


## Plot

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=5, highlight_topics=highlight_topics)

plt.savefig('output/top_slopes_plot.png')


nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=5, highlight_topics=highlight_topics,ascending=True)

plt.savefig('output/neg_slopes_plot.png')

