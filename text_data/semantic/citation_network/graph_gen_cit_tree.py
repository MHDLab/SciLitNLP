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
def set_round(G, round):
    """
    Set 'round' attribute of each node if it doesn't already exist
    """
    for node in G.nodes:
        if 'round' not in G.nodes[node]:
            G.nodes[node]['round'] = round
    return G

# regex = '%geographic information system%'
# ids = nu.fileio.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(25e6), output_limit=10000)
ids = pd.read_csv(r'C:\Users\aspit\Git\NLP\SciLitNLP\data_wrangling\semantic\text_analysis\data\indexed_searches.csv')['%energy storage%']
ids = ids.iloc[0:10000]
ids
#%%

def trim_graph(G, n_max):
    G = trim_graph_size(G, n_max*2)
    G = trim_graph_num_edges(G, 1)
    G = get_citation_info(G, con)
    G = trim_graph_fraction(G, n_max)
    G = trim_graph_num_edges(G, 1)
    return G

all_Gs = pd.Series(dtype='object')

G = nx.Graph()
G.add_nodes_from(ids)

## Initial internal citation trim
G = gen_citation_tree(G, con, cit_field='inCitations', add_new=False)
all_Gs['Initial'] = G.copy()
print("size of citation tree: {}".format(len(G.nodes)))

G = nu.citation.trim_connected_components(G, 1)

all_Gs['Keep largest connected graph'] = G.copy()

G = trim_graph(G, 300)
# G = get_citation_info(G, con)
# G = trim_graph_inCits(G, 300)
all_Gs['Trim to 300 most connected'] = G.copy()

#%%
#Grow citation graph including external citations
for i, n_max in enumerate([300,1000,3000,10000]):
    print('Growth Round {}'.format(i+1))
    G = set_round(G, i)
    G = gen_citation_tree(G, con, cit_field='both', add_new=True)
    print("size of final citation tree: {}".format(len(G.nodes)))
    G = trim_graph(G, n_max)

    G = nu.citation.trim_connected_components(G, 1)
    all_Gs['Growth Round {}, max {}'.format(i, n_max)] = G.copy()


G = set_round(G, i+1)

print("Loading final database")
df = load_df_semantic(con, G.nodes)

print("removing nodes not in database")
G.remove_nodes_from([id for id in G.nodes if id not in df.index])

nx.write_gexf(G, os.path.join('graphs','G_cit_tree.gexf'))
#%%

from fa2 import ForceAtlas2
forceatlas2 = ForceAtlas2()
#%%


fig, axes = plt.subplots(1,len(all_Gs), figsize=(3.5*len(all_Gs),4))

for i, round in enumerate(all_Gs.index):
    G_plot = all_Gs[round]
    # pos = nx.spring_layout(G_plot)
    pos = forceatlas2.forceatlas2_networkx_layout(G_plot, pos=None, iterations=25)
    nx.draw_networkx_nodes(G_plot, pos, ax=axes[i], node_size=10)
    nx.draw_networkx_edges(G_plot, pos, ax=axes[i],width=0.1)

    axes[i].set_title(round)


plt.savefig('output/cit_tree_rounds.png')

#%%

from bokeh.plotting import figure, from_networkx
from bokeh.io import output_notebook, show, output_file
from bokeh.models import (BoxSelectTool, Circle, EdgesAndLinkedNodes, HoverTool,
                          MultiLine, NodesAndLinkedEdges, Plot, Range1d, TapTool,Legend,LegendItem, ColumnDataSource)
from bokeh.palettes import Spectral4, Spectral6

cmap = Spectral6

rounds_plot = [0,1,2,3,4,5]
nodes_plot = [node for node in G.nodes if G.nodes[node]['round'] in rounds_plot]
G_plot = G.subgraph(nodes_plot)
G_plot = nx.Graph(G_plot)
# G_plot.remove_edges_from(G.edges) # Speed
print(len(G_plot.nodes))

plot = figure(plot_width=800, plot_height=600, output_backend='webgl')

#Dummy markers for legend, doesn't appear to be a way to have a colored legend with networx plot...
rounds = set(G_plot.nodes[node]['round'] for node in G_plot.nodes)
source = ColumnDataSource(dict(
    x = [0]*len(rounds),
    y = [0]*len(rounds),
    color = [cmap[r] for r in rounds],
    label=["Round {}".format(r) for r in rounds]
))

plot.circle(source=source, legend_group = 'label', color = 'color', visible=False)

# Networkx plot
for node in G_plot.nodes:
    # if node in df_2.index:
    G_plot.nodes[node]['title']= df.loc[node]['title']
    G_plot.nodes[node]['cit_round'] = cmap[G_plot.nodes[node]['round']]


pos = forceatlas2.forceatlas2_networkx_layout(G_plot, pos=None, iterations=50)
source = ColumnDataSource(dict(
x = [pos[key][0] for key in pos.keys()],
y = [pos[key][1] for key in pos.keys()],
color = [cmap[G_plot.nodes[node]['round']] for node in pos.keys()],
title = [G_plot.nodes[node]['title'] for node in pos.keys()],
))

plot.circle(source = source, color='color', radius = 3, radius_units='screen')

plot.add_tools(HoverTool(tooltips=[('title', '@title')]))

## from_networkx method doesn't seem to work with webgl
# graph = from_networkx(G_plot, lambda G: forceatlas2.forceatlas2_networkx_layout(G, iterations=10))
# graph.node_renderer.glyph = Circle(fill_color='cit_round')
# # graph.node_renderer.hover_glyph = Circle(size=15, fill_color='gray')
# # # graph.edge_renderer.glyph = MultiLine(line_color='black', line_alpha=0.4, line_width=1)
# graph.edge_renderer.visible = False
# plot.renderers.append(graph)

# output_notebook()
# show(plot)

output_file(filename='output/cit_tree_bokeh.html', title = 'Initial Citation Tree Rounds', )
show(plot)

#%%

con.close()