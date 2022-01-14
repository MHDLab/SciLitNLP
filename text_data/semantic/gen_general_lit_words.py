import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import sqlite3
import networkx as nx
from nlp_utils.fileio import load_df_semantic

db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

#%%
num_papers_gen_lit_sw = 10000
print('finding top terms in first {} papers for general literature stopwords'.format(num_papers_gen_lit_sw))

cursor.execute("SELECT id FROM raw_text LIMIT {}".format(num_papers_gen_lit_sw))
results = cursor.fetchall()
ids = [t[0] for t in results]

df = load_df_semantic(con, ids)

from nlp_utils.text_process import TextNormalizer
from nlp_utils.text_analysis import top_words

# The text we will analyze is words in both the title and abstract concatenated. 
docs = df['title'] + ' ' + df['paperAbstract']
texts = docs.apply(str.split)

text_normalizer = TextNormalizer()
texts_out = list(text_normalizer.transform(texts))
tw = top_words(texts_out,-1)

tw_word = [t[0] for t in tw]
tw_count = [t[1] for t in tw]

gen_lit_tw = pd.Series(tw_count, tw_word)
gen_lit_tw.to_csv('data/general_lit_top_words.csv')