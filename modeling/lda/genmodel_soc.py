import os
import sqlite3
import pandas as pd
import json
from nlp_utils.fileio import load_df_semantic
from lda_pipeline import lda_bigram_pipeline




#%%
fp_general_lit_tw = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/data/general_lit_top_words.csv')
gen_lit_tw = pd.read_csv(fp_general_lit_tw,index_col=0)
stopwords = gen_lit_tw[0:130].index.values


#%%
#Literature to analyze 

db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)
cursor = con.cursor()

# graph_data_folder = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/citation_network/graphs')
# G = nx.read_gexf(os.path.join(graph_data_folder, 'G_cit_tree.gexf'))
# df_tm = load_df_semantic(con, G.nodes)

fp_search_idx = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/data/indexed_searches.json')
with open(fp_search_idx, 'r') as f:
    id_dict = json.load(f)

idxs = id_dict['%energy storage%']
df_tm = load_df_semantic(con, idxs)
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']

#%%
texts = docs.apply(str.split)

fixed_bigrams = None
n_topics = 50
alpha = 1/n_topics
lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
bigram_kwargs={'threshold':20, 'min_count':10}

#%%

lda_model = lda_bigram_pipeline(texts, stopwords, fixed_bigrams, bigram_kwargs, lda_kwargs)
lda_model.idx = df_tm.index.values


if not os.path.exists('models'): os.mkdir('models')
lda_model.save(r'models/ldamod_soc.lda')