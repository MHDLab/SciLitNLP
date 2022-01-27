"""
Generate data associated with the LDA networkx (louvian clustering) plot. 
"""
import pandas as pd
import os
import sqlite3
import nlp_utils as nu
import argparse
from nlp_utils.fileio import load_df_semantic, load_df_SEAMs
import _pickle as cPickle
from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, choices = ['seams', 'soc'], help="Which type of dataset to use (previously generated)")
parser.add_argument('--min-edge-weight', type=float, default=0, help="minimum edge weight to be included in output graph")
args = parser.parse_args()

models_folder = os.path.join(os.getenv('REPO_DIR'), r'modeling/corex/models')

if args.dataset == 'soc':
    ## SOC
    model_path = os.path.join(models_folder, 'corexmod_soc.pkl')
    topic_model = cPickle.load(open(model_path, 'rb'))
    db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
    con = sqlite3.connect(db_path)
    df = load_df_semantic(con, topic_model.docs)
    df['inCitations'] = df['inCitations'].apply(",".join)
elif args.dataset == 'seams':
    # ## SEAMS
    model_path = os.path.join(models_folder, 'corexmod_seams.pkl')
    topic_model = cPickle.load(open(model_path, 'rb'))
    db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
    con = sqlite3.connect(db_path)
    df = load_df_SEAMs(con).dropna(subset=['OCR_text'])
    df['inCitations'] = ''
    edx_url_path = os.path.join(os.getenv('REPO_DIR'), 'text_data/seams/pdf_management/edx_urls.csv')
    df['display_url'] = pd.read_csv(edx_url_path, index_col=0).loc[df.index]

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)

df_topickeywords = s_topic_words.str.strip().str.split(' ', expand=True).astype(str)
df_topickeywords.columns = ['rank_{}'.format(i) for i in range(1,11)]
df_topickeywords.index.name = 'topic'

df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df.index , columns=s_topic_words.index)
df_doc_topic_probs = df_doc_topic_probs.div(df_doc_topic_probs.sum(axis=1), axis=0) # Normalize each topic...

df_edgekeywords, df_doc_edge_probs= nu.corex_utils.corex_edge_info(df_doc_topic_probs.values)
df_doc_edge_probs.index = df.index


if not os.path.exists('data'): os.makedirs('data')
df_topickeywords.to_csv(os.path.join('data','topic_keywords.csv'))
df_edgekeywords.to_csv(os.path.join('data','edge_keywords.csv'))#Generate graph of topic co-occurence

s_anchor = nu.corex_utils.get_s_topic_words(topic_model)
da_sigma, da_doc_topic = nu.corex_utils.calc_cov_corex(topic_model, s_anchor.index, topic_model.docs.values)
#%%

from pipeline_data_prep import pipeline_data_prep
pipeline_data_prep(df, df_topickeywords, df_doc_topic_probs, df_doc_edge_probs, da_sigma, min_edge_weight = args.min_edge_weight)

