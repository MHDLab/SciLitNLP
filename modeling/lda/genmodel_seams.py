import os
import sqlite3
from nlp_utils.fileio import load_df_SEAMs
from lda_pipeline import lda_bigram_pipeline


db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
con = sqlite3.connect(db_path)
df_tm = load_df_SEAMs(con).dropna(subset=['OCR_text'])
docs = df_tm['title'] + ' ' + df_tm['OCR_text']

stopwords=[]


texts = docs.apply(str.split)

fixed_bigrams = None
n_topics = 50
alpha = 1/n_topics
lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
bigram_kwargs={'threshold':20, 'min_count':10}

lda_model = lda_bigram_pipeline(texts, stopwords, fixed_bigrams, bigram_kwargs, lda_kwargs)
lda_model.idx = df_tm.index.values


if not os.path.exists('models'): os.mkdir('models')
lda_model.save(r'models/ldamod_seams.lda')