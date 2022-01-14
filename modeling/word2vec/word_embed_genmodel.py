"""
This script generates a word2vec model using gensim.
"""

#%%
from gensim.models import Word2Vec

import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import sqlite3
import nlp_utils as nu

db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

ids = pd.read_csv(r'C:\Users\aspit\Git\NLP\SciLitNLP\text_data\semantic\text_analysis\data\indexed_searches.csv')['%energy storage%']
df_tm = nu.fileio.load_df_semantic(con, ids)

docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

fp_general_lit_tw = r'C:\Users\aspit\Git\NLP\SciLitNLP\text_data\semantic\text_analysis\data\general_lit_top_words.csv'
gen_lit_tw = pd.read_csv(fp_general_lit_tw,index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values
fixed_bigrams = []

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

texts_bigram = pipeline.fit_transform(texts)

#%%
from nlp_utils import text_analysis

tw = text_analysis.top_words(texts, num_words=50)

print([w[0] for w in tw])

# list(sentences)
# %%
mod = Word2Vec(sg=0, seed=42, min_count = 100, window= 5, vector_size=100, negative = 15)

#%%
mod.build_vocab(texts)

#%%
print("Training Model")
mod.train(texts, total_examples = mod.corpus_count, epochs=30)
# %%

# mod.wv.init_sims()

mod.top_words = [w[0] for w in tw]

if not os.path.exists('w2v_models'): os.makedirs('w2v_models')

mod.save(r"models/word2vec_semantic.model")
# %%
