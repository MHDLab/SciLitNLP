import pandas as pd
import os
import json
import sqlite3
import networkx as nx
import argparse
import nlp_utils as nu
from corex_pipeline import corex_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices = ['search', 'cit_tree'], default= 'search',help="Which type of dataset to use (previously generated)")
parser.add_argument('-r', '--regex', type=str, default = '', help="regular expression generated from indexed search (if using indexed search id generation)")
parser.add_argument('-n', '--n_topics', type=int, default = 20, help="Number of topics")
parser.add_argument('-tw', '--remove-top-words', action='store_true', help="Use general literature data stopwords")

args = parser.parse_args()

model_dict = vars(args)
model_dict['model_type'] = 'CorEx'

if args.remove_top_words:
    print("using general literature stopwords")
    fp_general_lit_tw = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/data/general_lit_top_words.csv')
    gen_lit_tw = pd.read_csv(fp_general_lit_tw,index_col=0)
    stopwords = gen_lit_tw[0:130].index.values
else:
    print("Not using general literature stopwords")
    stopwords = []

db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)
cursor = con.cursor()

if args.dataset == 'cit_tree':
    graph_data_folder = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/citation_network/graphs')
    G = nx.read_gexf(os.path.join(graph_data_folder, 'G_cit_tree.gexf'))
    df_tm = nu.fileio.load_df_semantic(con, G.nodes)

elif args.dataset == 'search':

    fp_search_idx = os.path.join(os.getenv('REPO_DIR'), r'text_data/semantic/data/indexed_searches.json')
    with open(fp_search_idx, 'r') as f:
        id_dict = json.load(f)

    if not args.regex in id_dict:
        raise ValueError("Search term {} not found in indexed searches".format(args.regex))

    idxs = id_dict[args.regex]
    df_tm = nu.fileio.load_df_semantic(con, idxs)


else:
    raise ValueError("Didn't get valid dataset type ") # Shouldn't happen with choices in parseargs...

model_dict['n_papers'] = len(df_tm)

docs = df_tm['title'] + ' ' + df_tm['paperAbstract']

#Topic modeling
print("Corex Topic Modeling")

corex_anchors = []
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)
n_hidden = args.n_topics

topic_model = corex_pipeline(docs, stopwords, corex_anchors, fixed_bigrams, n_hidden)

topic_model.pipeline_settings = model_dict

import _pickle as cPickle
#Save model
if not os.path.exists('models'): os.mkdir('models')
cPickle.dump(topic_model, open(os.path.join('models', 'corexmod_soc.pkl'), 'wb'))