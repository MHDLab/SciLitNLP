"""
This script generates data associated with the wedge plot.
Note the nature of this plot requires that an LDA model with 20 topics or less is used (number of available colors)
"""
#%%
import sqlite3
import pandas as pd
import os
import sys
import numpy as np
from gensim.models import LdaModel

from nlp_utils import gensim_utils, sklearn_utils, fileio


# db_folder = r'E:\\'
# con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
# regex = '%energy storage%'
# ids = fileio.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e6), output_limit=int(3e4))
# df = fileio.load_df_semantic(con, ids)
# docs = df['title'] + ' ' + df['paperAbstract']

#%%

db_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs'
df = fileio.load_df_SEAMs(db_folder).dropna(subset=['processed_text'])
# df = df.sample(300, random_state=42)
docs = df['processed_text']
texts = docs.apply(str.split)

#%%


#Load in LDA model (should match text data)

print("LDA Topic Modeling")

import nlp_utils as nu
from nlp_utils.gensim_utils import basic_gensim_lda
from nlp_utils import gensim_utils
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

fixed_bigrams = []

pipeline = Pipeline([
('text_norm', nu.text_process.TextNormalizer()),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

texts_bigram = pipeline.fit_transform(texts)

n_topics = 20
alpha = 1/n_topics

lda_kwargs = {'alpha': alpha, 'eta': 0.03, 'num_topics':n_topics, 'passes':5}
id2word, data_words, lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)

#%%

print('Generating topic probability matrix')

df_topickeywords, df_doc_topic_probs= gensim_utils.gensim_topic_info(lda_model, data_words, id2word)
df_doc_topic_probs.index = df.index

#%%
df_meta = pd.DataFrame(index=df.index)

print("Generating Tsne")

tsne_x, tsne_y = sklearn_utils.calc_tsne(texts_bigram)

df_meta['tsne_x'] = tsne_x
df_meta['tsne_y'] = tsne_y


display_text = " <a href=" + df['display_url'] + ">" + df['title'] + "</a> <br>"

df_meta['display_text'] = display_text

# %%
print('Generating Wedges')

def gen_wedges(df_subset, num_wedges_per_paper):
    """
    input: dataframe of topic probabilities for each paper

    Calculates wedge angles from the top num_wedges_per_paper topics for each paper

    returns: dataframe indexed by [topic, ID] with topic probability, and corresponding slice_start and slice_end
    """

    all_dfs = []
    for index, row in df_subset.iterrows():
        
        top_topics = row.argsort()[::-1][0:num_wedges_per_paper]
        top_topics_prob = row.iloc[top_topics]
        top_topics_prob.name = 'top_topics_prob'
            #Normalize ,    
        # top_topics_prob = top_topics_prob/top_topics_prob.sum()
        slice_start = []
        slice_end = []
        for i, topic in enumerate(top_topics_prob.index):
            prob = top_topics_prob[topic] 
            if i == 0:
                slice_start.append(0)
                slice_end.append(prob)
            else:
                slice_start.append(slice_end[i-1])
                slice_end.append(slice_end[i-1] + prob)
        slice_start = pd.Series(slice_start, index=top_topics_prob.index)*(-2*np.pi)
        slice_start.name = 'slice_start'
        slice_end = pd.Series(slice_end, index=top_topics_prob.index)*(-2*np.pi)
        slice_end.name = 'slice_end'
        df_slices = pd.concat([top_topics_prob, slice_start, slice_end], axis=1)
        all_dfs.append(df_slices)

    df_wedges = pd.concat(all_dfs, keys=df_subset.index, names = ['id', 'Topic'])
    df_wedges = df_wedges.swaplevel()
    df_wedges = df_wedges.sort_index(level=0)
    return df_wedges

num_wedges_per_paper = 3

df_wedges = gen_wedges(df_doc_topic_probs, num_wedges_per_paper)

df_wedges

#%%

#Calculate strings expressing the top topics for each paper
top_topics = []
for id, row in df_doc_topic_probs.iterrows():
    top_topic = row.argsort()[::-1][0]
    top_topics.append(df_doc_topic_probs.columns[top_topic])

df_meta['top_topic'] = top_topics

#%%
top_topics_prob = df_wedges.swaplevel().sort_index(level=0)['top_topics_prob']
top_topics_prob

IDs = top_topics_prob.index.levels[0]
top_topics_str = pd.Series(index=IDs)

for ID in IDs:
    temp = top_topics_prob.loc[ID]
    temp = temp.sort_values(ascending=False)
    tempstr = ''
    for topic in temp.index:
        tempstr += str(topic) + ': ' + str(int(temp[topic]*100)) + '%, '
    top_topics_str.loc[ID] = tempstr

df_meta['top_topics_str'] = top_topics_str

#%%

if not os.path.exists('data'): os.mkdir('data')

df_meta.to_csv(r'data\df_meta.csv')
df_doc_topic_probs.to_csv(r'data\df_doc_topic_probs.csv')
df_wedges.to_csv(r'data\df_wedges.csv')
df_topickeywords.to_csv(r'data\df_topickeywords.csv')