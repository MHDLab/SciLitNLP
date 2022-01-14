"""
Generate data associated with the LDA networkx (louvian clustering) plot. 
"""
#%%

import pandas as pd
import xarray as xr
import numpy as np
import os
import sys
import matplotlib.pyplot as plt 
from gensim.models import LdaModel
from nlp_utils import gensim_utils, sklearn_utils, fileio
from nlp_utils.fileio import load_df_semantic

import _pickle as cPickle

model_path = os.path.join(r'C:\Users\aspit\Git\NLP\modeling\corex\models', 'mod_cit_tree.pkl')
topic_model = cPickle.load(open(model_path, 'rb'))

db_folder = r'E:\\'
import sqlite3
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

df = load_df_semantic(con, topic_model.docs.values)
df = df.rename({'year': 'Year', 's2Url': 'display_url'}, axis=1)
df['inCitations'] = df['inCitations'].apply(",".join)
df_tm = df

import nlp_utils as nu

s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)

df_topickeywords = s_topic_words.str.strip().str.split(' ', expand=True).astype(str)
df_topickeywords.columns = ['rank_{}'.format(i) for i in range(1,11)]
df_topickeywords.index.name = 'topic'

df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_doc_topic_probs = df_doc_topic_probs.div(df_doc_topic_probs.sum(axis=1), axis=0) # Normalize each topic...

df_edgekeywords, df_doc_edge_probs= nu.corex_utils.corex_edge_info(df_doc_topic_probs.values)
df_doc_edge_probs.index = df.index


data_folder = 'data'
if not os.path.exists(data_folder): os.makedirs(data_folder)
df_topickeywords.to_csv(os.path.join(data_folder,'topic_keywords.csv'))
df_edgekeywords.to_csv(os.path.join(data_folder,'edge_keywords.csv'))
#%%

df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['Year'], norm_each_topic=False)
df_topicsyear.to_csv(os.path.join(data_folder,'topic_year.csv'))

# %%

top_papers_topic = pd.Series(index= df_doc_topic_probs.columns)
top_papers_topic.index.name = 'topic'
top_topic_paper = df_doc_topic_probs.idxmax(axis=1)
top_papers_6_10 = pd.Series(index= df_doc_topic_probs.columns)
top_papers_6_10.index.name = 'topic'

for topic in df_doc_topic_probs.columns:
    
    #first part of string
    top_papers = df_doc_topic_probs[topic].sort_values(ascending=False)
    text = '<h3>Papers with highest probability for selected topic: </h3>'

    for idx in top_papers[0:5].index:
        prob = top_papers[idx]
        # idx = str(idx)

        linkstr = df['title'][idx] + " (" + str(df['Year'][idx]) + ")"
        text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
        text += " (topic prob: {:0.1f}%)".format(prob*100)
        text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
        text += " <br><br> "
    
    top_papers_topic[topic] = text


    #
    papers = top_topic_paper[top_topic_paper == topic].index
    # log_probs = df.loc[papers]['logprob'].sort_values(ascending=False)[0:5]
    text = '<h3> </h3>'

    for idx in top_papers[6:11].index:

        prob = top_papers[idx]

        linkstr = df['title'][idx] + " (" + str(df['Year'][idx]) + ")"
        text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
        text += " (topic prob: {:0.1f}%)".format(prob*100)
        text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
        text += " <br><br> "

    # top_papers_str = ", ".join(df['title'][top_papers.index])
    
    top_papers_6_10[topic] = text


top_papers_topic.to_csv(os.path.join(data_folder,'top_papers_topic.csv'))
top_papers_6_10.to_csv(os.path.join(data_folder,'top_papers_6_10.csv'))


top_papers_edge= pd.Series(index= df_doc_edge_probs.columns)
top_edge_paper = df_doc_edge_probs.idxmax(axis=1)
edge_papers_6_10 = pd.Series(index= df_doc_edge_probs.columns)

loop_control = 0
for edge in df_doc_edge_probs.columns:
    
    #first part of string
    top_papers = df_doc_edge_probs[edge].sort_values(ascending=False)
    text = '<h3>Papers with highest probability for selected edge: </h3>'

    for idx in top_papers[0:5].index:
        prob = top_papers[idx]

        linkstr = df['title'][idx] + " (" + str(df['Year'][idx]) + ")"
        text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
        text += " (topic prob: {:0.1f}%)".format(prob*100)
        text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
        text += "<br><br> "
    
    top_papers_edge[edge] = text

    #
    papers = top_edge_paper[top_edge_paper == edge].index
    # log_probs = df.loc[papers]['logprob'].sort_values(ascending=False)[0:5]
    text = '<h3> </h3>'

    for idx in top_papers[6:10].index:
        prob = top_papers[idx]
        
        linkstr = df['title'][idx] + " (" + str(df['Year'][idx]) + ")"
        text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
        text += " (topic prob: {:0.1f}%)".format(prob*100)
        text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
        text += "<br><br> "

    # top_papers_str = ", ".join(df['title'][top_papers.index])
    
    edge_papers_6_10[edge] = text
    
top_papers_edge.to_csv(os.path.join(data_folder,'top_papers_edge.csv'))
edge_papers_6_10.to_csv(os.path.join(data_folder,'edge_papers_6_10.csv'))
# %%

#Generate graph of topic co-occurence

s_anchor = nu.corex_utils.get_s_topic_words(topic_model)
da_sigma, da_doc_topic = nu.corex_utils.calc_cov_corex(topic_model, s_anchor.index, topic_model.docs.values)
print('Generating Graph')

import networkx as nx 
G = nx.Graph() 

min_edge_weight = 1 #Minimum edge weight to be included.

for topic_i in da_sigma.coords['topic_i'].values:
    for topic_j in da_sigma.coords['topic_j'].values:
        weight = da_sigma.sel(topic_i=topic_i, topic_j=topic_j).item()
        if  weight > min_edge_weight:  
            if topic_i != topic_j:
                G.add_edge(topic_i,topic_j, weight=weight)

topic_keywords = df_topickeywords[df_topickeywords.columns[0:6]].apply(", ".join, axis= 1)

for node in G.nodes:
    G.nodes[node]['disp_text'] = topic_keywords[node].replace(',', '\n')
    G.nodes[node]['size'] = df_doc_topic_probs.sum()[node]/10

#%%

print('Writing graph to disk')

nx.write_gexf(G, os.path.join(data_folder,'G_topic.gexf'))
# %%
