import pandas as pd
import os
import json
import sqlite3
import nlp_utils as nu
from corex_pipeline import corex_pipeline

fp_general_lit_tw = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/data/general_lit_top_words.csv')
gen_lit_tw = pd.read_csv(fp_general_lit_tw,index_col=0)
stopwords = gen_lit_tw[0:130].index.values


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
df_tm = nu.fileio.load_df_semantic(con, idxs)
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']



print("Corex Topic Modeling")

corex_anchors = []
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)

topic_model = corex_pipeline(docs, stopwords, corex_anchors, fixed_bigrams)

import _pickle as cPickle
#Save model
if not os.path.exists('models'): os.mkdir('models')
cPickle.dump(topic_model, open(os.path.join('models', 'corexmod_soc.pkl'), 'wb'))