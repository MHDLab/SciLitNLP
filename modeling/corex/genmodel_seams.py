import pandas as pd
import os
import json
import sqlite3
import nlp_utils as nu
from corex_pipeline import corex_pipeline

db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
con = sqlite3.connect(db_path)
df_tm = nu.fileio.load_df_SEAMs(con).dropna(subset=['OCR_text'])
docs = df_tm['title'] + ' ' + df_tm['OCR_text']

stopwords=[]

corex_anchors = ['corros','anti_corros', 'corrosion_resist']
fixed_bigrams = nu.corex_utils.anchors_to_fixed_bigrams(corex_anchors)

topic_model = corex_pipeline(docs, stopwords, corex_anchors, fixed_bigrams)

import _pickle as cPickle
#Save model
if not os.path.exists('models'): os.mkdir('models')
cPickle.dump(topic_model, open(os.path.join('models', 'corexmod_seams.pkl'), 'wb'))