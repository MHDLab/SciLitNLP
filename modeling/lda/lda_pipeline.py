import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import os
import sqlite3
import networkx as nx
from nlp_utils.fileio import load_df_semantic
import nlp_utils as nu
from sklearn.pipeline import Pipeline
import json
#%%

db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

#%%
fp_general_lit_tw = r'C:\Users\aspit\Git\NLP\SciLitNLP\text_data\semantic\data\general_lit_top_words.csv'
gen_lit_tw = pd.read_csv(fp_general_lit_tw,index_col=0)
gen_lit_remove = gen_lit_tw[0:130].index.values


#%%
#Literature to analyze 
# graph_data_folder = r'C:\Users\aspit\Git\NLP\SciLitNLP\text_data\semantic\citation_network\graphs'
# G = nx.read_gexf(os.path.join(graph_data_folder, 'G_cit_tree.gexf'))
# df_tm = load_df_semantic(con, G.nodes)

fp_search_idx = r'C:\Users\aspit\Git\NLP\SciLitNLP\text_data\semantic\data\indexed_searches.json'
with open(fp_search_idx, 'r') as f:
    id_dict = json.load(f)

idxs = id_dict['%carbon nanotube%']
df_tm = load_df_semantic(con, idxs)

#%%
docs = df_tm['title'] + ' ' + df_tm['paperAbstract']
texts = docs.apply(str.split)

fixed_bigrams = None

print("LDA Topic Modeling")
from nlp_utils.gensim_utils import basic_gensim_lda

pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=gen_lit_remove)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

texts_bigram = pipeline.fit_transform(texts)

n_topics = 30
alpha = 1/n_topics

lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)


lda_model.texts_bigram = texts_bigram
lda_model.id2word = id2word
lda_model.data_words = data_words
lda_model.idx = df_tm.index.values
lda_model.save(r'models\ldamod_cit_tree.lda')
