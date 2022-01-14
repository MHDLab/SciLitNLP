#%%
from community.community_louvain import partition_at_level
from networkx.algorithms.centrality import load
import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.fileio import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph_fraction, get_citation_info, trim_graph_inCits, trim_graph_num_edges, trim_graph_size

from fa2 import ForceAtlas2
forceatlas2 = ForceAtlas2()

db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()


#%%
#TODO: this doesn't seem to work well with broad 

# regex = '%energy storage%'
# ids = nu.fileio.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(30e6), output_limit=30e6)
# pd.Series(ids, name=regex.strip('%')).to_csv('data/indexed_search.csv')

ids = pd.read_csv('data/indexed_search.csv', index_col=0)['energy storage']

G = nx.Graph()
G.add_nodes_from(ids)

G = gen_citation_tree(G, con, cit_field='inCitations', add_new=False)
# print("size of citation tree: {}".format(len(G.nodes)))
# nx.write_gexf(G, 'data/G_energystorage.gexf')
#%%
num_cc = 1
cc= pd.Series(nx.connected_components(G))
cc_len = cc.apply(len).sort_values(ascending=False)
cc_keep = cc.loc[cc_len.iloc[0:num_cc].index]
ids_keep = set(set().union(*cc_keep.values))

G2 = G.subgraph(ids_keep)
#%%
import community as community_louvain

partition = community_louvain.best_partition(G2, resolution=2)

print("number partitions: {}".format(max(partition.values())))

#%%
import matplotlib.cm as cm
pos = forceatlas2.forceatlas2_networkx_layout(G2, pos=None, iterations=20)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(G2, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(G2, pos, alpha=0.5)
plt.show()

#%%

Gsubs = []

for p in set(partition.values()):
    plist = [i for i in partition.keys() if partition[i] == p]

    Gsub = G.subgraph(plist)

    Gsub = get_citation_info(Gsub, con)
    Gsub = nu.citation.trim_graph_fraction(Gsub, 1000)

    Gsub = gen_citation_tree(Gsub, con, cit_field='both', add_new=True)

    Gsub = trim_graph_size(Gsub,20000)
    Gsub = get_citation_info(Gsub, con)
    Gsub = nu.citation.trim_graph_fraction(Gsub, 5000)

    Gsubs.append(Gsub)

#%%

# for Gsub in Gsubs:
#     df_sub = load_df_semantic(con, Gsub.nodes)

#     textnorm = nu.text_process.TextNormalizer()
#     corpus = (df_sub['title'] + ' ' + df_sub['paperAbstract']).str.split()
#     corpus  = textnorm.fit_transform(corpus)
#     print(nu.text_analysis.top_words(corpus))


#%%

all_nodes = []
for Gsub in Gsubs:
    all_nodes.extend(Gsub.nodes)

# df_final = load_df_semantic(con, all_nodes)

G_final = nx.Graph()
G_final.add_nodes_from(set(all_nodes))

nx.write_gexf(G, os.path.join('graphs','G_cit_tree.gexf'))

#%%
