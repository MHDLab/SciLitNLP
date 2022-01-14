#%%
from bokeh.models.renderers import GraphRenderer
from bokeh.models.sources import ColumnDataSource
import pandas as pd
import xarray as xr
import os
import sqlite3
import networkx as nx

import matplotlib.pyplot as plt
from nlp_utils.fileio import load_df_semantic
import nlp_utils as nu

from nlp_utils.citation import gen_citation_tree, trim_graph_fraction, get_citation_info, trim_graph_inCits, trim_graph_num_edges, trim_graph_size

db_folder = r'E:\\'
con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
cursor = con.cursor()

#%%

# regex = '%energy storage%'
# ids = nu.fileio.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(5e6), output_limit=3000)


ids = pd.read_csv('data/indexed_search.csv', index_col=0)['energy storage']

G = nx.Graph()
G.add_nodes_from(ids)

G = gen_citation_tree(G, con, cit_field='both', add_new=True)
print("Citation tree size: {}".format(len(G.nodes)))

G = trim_graph_size(G,100000)
G = get_citation_info(G, con)
G = nu.citation.trim_graph_fraction(G, 50000)

nx.write_gexf(G, os.path.join('graphs','G_cit_tree.gexf'))
# %%
