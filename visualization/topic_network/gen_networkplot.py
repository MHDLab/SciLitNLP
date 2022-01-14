# %%
"""
Generate the networkx louvian clustering plot html file. The louvian communities are calculated in this script. 
"""


from re import I
import sys
from bokeh.core.property.primitive import Null
from networkx.algorithms.centrality import group
from networkx.algorithms.minors import contracted_nodes
import pandas as pd
import numpy as np

from community import community_louvain
import matplotlib.cm as cm
import networkx as nx

import os
data_folder = 'data'

import sys



# Generate Louvian Communities
#load graph
G = nx.read_gexf(os.path.join(data_folder,'G_topic.gexf'))

if len(sys.argv) > 1:
    resolution = float(sys.argv[1])
else:
    resolution = 1
    
partition = community_louvain.best_partition(G, resolution = resolution, random_state=2)
print('Number Partitions ' + str(list(set(partition.values()))))

#get colormap
cmap = cm.get_cmap('Set1', max(partition.values())+1 )

# square root scaling of edge size and weight to bring out smaller topics a bit.  
for node in G.nodes:
    G.nodes[node]['size'] = np.sqrt(G.nodes[node]['size'])*3

for edge in G.edges:
    G.edges[edge]['weight'] = np.sqrt(G.edges[edge]['weight'])*1.5


#%%

#Reduce topic index/columns to those present in the network 
present_topics = list(G.nodes)
present_edges = list(e[0]+', '+e[1] for e in G.edges)

df_topickeywords = pd.read_csv(os.path.join(data_folder, 'topic_keywords.csv'), index_col=0).loc[present_topics]
df_topic_year = pd.read_csv(os.path.join(data_folder, 'topic_year.csv'), index_col=0)[present_topics]

df_topic_year = df_topic_year.loc[:2020]
top_papers_topic = pd.read_csv(os.path.join(data_folder, 'top_papers_topic.csv'), index_col=0, squeeze=True)[present_topics]
top_papers_6_10 = pd.read_csv(os.path.join(data_folder, 'top_papers_6_10.csv'), index_col=0, squeeze=True)[present_topics]

#Reduce edges to those present in graph
df_edgekeywords = pd.read_csv(os.path.join(data_folder, 'edge_keywords.csv'), index_col=0).loc[present_edges]
top_papers_edge = pd.read_csv(os.path.join(data_folder, 'top_papers_edge.csv'), index_col=0, squeeze=True)[present_edges]
edge_papers_6_10 = pd.read_csv(os.path.join(data_folder, 'edge_papers_6_10.csv'), index_col=0, squeeze=True)[present_edges]

#calculate the probability over the past 5 years
recent_prob = df_topic_year.loc[2015:2020].mean()
recent_prob = recent_prob/recent_prob.max()

# function for adjusting color intensity as a function of the probability over the past 5 years
def change_intensity(colors, probs):
    for c in range(len(colors)):
        colors[c] = colors[c].strip('#')
        lv = len(colors[c])
        colors[c] =tuple(int(((.45 + .8*probs[c]) * int(colors[c][i:i + lv //3], 16))) for i in range(0, lv, lv//3))
        colors[c] = '#%02x%02x%02x' % colors[c]
    return colors

#%%
# import bokeh things
from bokeh.models.sources import ColumnDataSource, CustomJS
from bokeh.io import output_file, show, output_notebook
from bokeh.models import (Circle, HoverTool,
                          MultiLine, Range1d, TapTool, Div, Circle, MultiLine, Legend)
from bokeh.plotting import figure, show, from_networkx, save
from bokeh.palettes import Colorblind3, Colorblind4, Colorblind5, Colorblind6, Colorblind7
from bokeh.layouts import column, row, layout

intro_text = Div(text="""
<h2> Graph Key: </h2>
<h3> Each node represents a topic and each edge between two topics represents the likelihood of the two topics appearing in an abstract together.<br>
<b>NODE COLOR:</b> community assigned by the Louvain community detection algorithm.<br>
<b>NODE SIZE:</b> overall probability of the topic appearing in abstracts.<br>
<b>NODE BRIGHTNESS:</b> probability of the topic appearing in abstracts over the past 5 years (brighter = higher probability).<br>
<b>EDGE THICKNESS:</b> probability of two topics occurring in the same abstract.<br>
(Note, graph projection is designed to make it easier to see connections between nodes.  The axes do not communicate additional information.)<br><br>
You can select different topics to pull up additional information including the probability of abstracts containing topics over time and papers with the highest probability of containing the selected topic.<br>
Click on a topic to view further information, or select two topic nodes to view information about their relationship.</h3>
""", width=1200)
trend_text = Div(text="""<h3>The Topic Probability Trend shows the probability of an abstract from the corpus containing the selected topic during a give year.</h3>""")
group_text = Div(text="""<h3>The graph below uses the same data from the graph above, but merges nodes from the same Louvain community.</h3>""")
table_text = Div(text="""<br><br><br><br><h3>The table below shows the topics, their keywords, their probabilities, and their Louvain communities.<br>
They are grouped by community</h3>""", height=150)

source_topics_year = ColumnDataSource(df_topic_year)

#pallet dictionary
#pal_dict = {3:Colorblind3, 4 : Colorblind4, 5 : Colorblind5,  6: Colorblind6, 7:Colorblind7}
#https://medium.com/sketch-app-sources/mixing-colours-of-equal-luminance-part-2-3e10c07c947c
#green, blue, seagreen, yellow, purple, rust, orange
custom7 = ('#32C855','#3278C8','#32C8AA','#C8C832','#7832C8','#C85032', '#C89132')
# green, orange, blue, yellow, purple, rust
custom6 = ('#32C855','#C89132','#3278C8','#C8C832','#7832C8', '#C85032')
# rust, orange, blue, purple, green
custom5 = ('#C85032','#C89132','#3278C8','#7832C8', '#32C855')
#green, blue, orange, purple
custom4 = ('#32C855','#3278C8','#C89312','#7832C8')
#orange, blue, green
custom3 = ('#C89132','#3278C8','#32C855')

pal_dict = {3:custom3, 4:custom4, 5:custom5, 6:custom6, 7:custom7}

pal = pal_dict[len(set(partition.values()))]
#calculate the colors based on partition (Louvian community) and recent probability
fill_colors = [pal[i] for i in partition.values()]
fill_colors = change_intensity(fill_colors, recent_prob)
#generate the plot
plot = figure(plot_width=1200, plot_height=850,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
#add a hover tool for nodes
node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("text", "@disp_text")])
plot.add_tools(node_hover_tool,TapTool())
plot.axis.visible = False

#render the graph
graph_renderer = from_networkx(G, nx.kamada_kawai_layout, scale=1, center=(0, 0))

#add node data to the source data
source_graph = graph_renderer.node_renderer.data_source
source_graph.add(fill_colors, 'color')
source_graph.add(top_papers_topic, 'top_papers_topic')
source_graph.add(top_papers_6_10, 'top_papers_6_10')
graph_renderer.node_renderer.glyph = Circle(radius = 'size', fill_color = 'color', radius_units='screen')

#add edge data to the source data
source_edges = graph_renderer.edge_renderer.data_source
edge_colors = ['#444444']*len(source_edges.data['id'])
source_edges.add(edge_colors, 'color')
graph_renderer.edge_renderer.glyph = MultiLine(line_width = 'weight', line_alpha = 0.35, line_color='color')

#divs for text 
topic_words_list = Div(text="""""")
text_top_papers_topic = Div(text="""""")
text_top_papers_6_10 = Div(text="""""")
topic_words_list_2 = Div(text="""""")
top_papers_topic_2 = Div(text="""""")
top_papers_6_10_2 = Div(text="""""")

#plot line graph of paper years and topics
years = df_topic_year.index

#data source for trend info
source_trend = ColumnDataSource(data=dict(year=years, prob=[0]*len(years), comp_prob=[0]*len(years)))
#plot for trend info
plot_trend = figure(plot_width=700, plot_height=400, title = 'Topic Probability Trend')
plot_trend.xaxis.axis_label = 'Year'
plot_trend.yaxis.axis_label = 'Normalized Topic Probability'

#trend lines for first and second topics
plot_trend_line = plot_trend.line(source=source_trend, x='year', y = 'prob', line_width=2, line_color='#D86EFF')
comp_trend_line = plot_trend.line(source=source_trend, x='year', y = 'comp_prob', line_width=2, line_color='#8B6EFF')
trend_legend = Legend(items=[('', [plot_trend_line]), ('', [comp_trend_line])])
plot_trend.add_layout(trend_legend, 'right')
#make plots invisible until nodes are selected
plot_trend_line.visible = False
comp_trend_line.visible = False

###Javascript Callback for selecting nodes

topic_part = pd.Series(list(partition.values()), index= G.nodes)
# keep track of whatever the most recently selected node was
prev_id = ColumnDataSource(data={'prev': [-1]})

# callback args is a dictionary of variables that are used into the javascript callback
callback_args = dict(
    source_trend = source_trend,
    source_graph = source_graph,
    source_edges = source_edges,
    source_topics_year=source_topics_year,
    topic_words_list=topic_words_list,
    topic_words_list_2 = topic_words_list_2,
    text_top_papers_topic=text_top_papers_topic,
    top_papers_topic_2 = top_papers_topic_2,
    text_top_papers_6_10=text_top_papers_6_10,
    top_papers_6_10_2 = top_papers_6_10_2,
    topic_keywords=df_topickeywords[df_topickeywords.columns[0:6]].apply(", ".join, axis= 1)[present_topics],
    edge_keywords=df_edgekeywords[df_edgekeywords.columns[0:6]].apply(", ".join, axis= 1)[present_edges],
    edge_list = present_edges,
    top_papers_edge = top_papers_edge,
    edge_papers_6_10 = edge_papers_6_10,
    plot_trend_line =plot_trend_line,
    comp_trend_line = comp_trend_line,
    trend_legend = trend_legend,
    prev_id = prev_id
 )

# Javascript callback for when you select nodes
js_callback = CustomJS(args=callback_args, code = """

//ids is technically a list the indices of selected nodes, but there should only ever be one
var ids = source_graph.selected.indices;
console.log(topic_keywords);

// A function for updating the text with relevant information to the selected nodes and edges
function updateText(topic_words, topic_words_2, top_papers, top_papers_2, top_6_10, top_6_10_2) {

        topic_words_list.text = topic_words; 
        topic_words_list_2.text = topic_words_2; 
        topic_words_list.change.emit();
        topic_words_list_2.change.emit();

        text_top_papers_topic.text = top_papers;
        top_papers_topic_2.text = top_papers_2;
        text_top_papers_topic.change.emit();
        top_papers_topic_2.change.emit();

        text_top_papers_6_10.text = top_6_10;
        top_papers_6_10_2.text = top_6_10_2;
        text_top_papers_6_10.change.emit();
        top_papers_6_10_2.change.emit();

}

// if any nodes are selected, highlight and print stuff
if (ids.length > 0){

    //get the topic and the id of the previously selected node
    var topic = source_graph.data['index'][ids[0]];
    var p_id = prev_id.data['prev'][0];
    
    console.log(p_id);
    console.log(topic);

    // if only one node is selected, highlight the node + edges and display topic info for that node
    if (p_id < 0) {

        //highlight the edges connected to selected node
        for (var i=1; i<source_edges.data['start'].length; i++){
            if (source_edges.data['start'][i] == topic || source_edges.data['end'][i] == topic){
                source_edges.data['color'][i] = '#AB52FF';
            } else{
                source_edges.data['color'][i] = '#888888';
            }
        }
        source_edges.change.emit();

        //print topic related info
        updateText(
            ('<h2>'+topic+' key_words: '+topic_keywords[ids[0]]+'</h2>'),
             '', 
             source_graph.data['top_papers_topic'][ids[0]], 
             '', 
             source_graph.data['top_papers_6_10'][ids[0]],
             ''
        )

        //plot trend line
        var data_trend = source_topics_year.data[topic];
        source_trend.data['prob'] = data_trend;
        trend_legend.items[0].label.value = topic;
        plot_trend_line.visible = true;
        source_trend.change.emit();

        //if two nodes are selected, highlight both their edges, especially the edge connecting them and print the edge information
        } else {
        
        //add the previous id to selected indices so both nodes are at full opacity
        source_graph.selected.indices = [ids[0], p_id];
        var prev_topic = source_graph.data['index'][p_id];
        
        //highlight the edges connected to each node and make the edge between them extra dark
        var no_edge_found = true;
        for (var i=1; i<source_edges.data['start'].length; i++){
            if ((source_edges.data['start'][i] == topic && source_edges.data['end'][i] == prev_topic) || (source_edges.data['start'][i] == prev_topic && source_edges.data['end'][i] == topic)){
                source_edges.data['color'][i] = '#8503FF';
                no_edge_found = false;
            } else if (source_edges.data['start'][i] == topic || source_edges.data['end'][i] == topic || source_edges.data['start'][i] == prev_topic || source_edges.data['end'][i] == prev_topic){
                source_edges.data['color'][i] = '#A38FFF';
            } else {
                source_edges.data['color'][i] = '#888888';
            }
        }

        // if there exists no edge between the two nodes, print the information for both topics
        if (no_edge_found){
            console.log('no edge between these two nodes');

            updateText(
                ('No edge exists between the two selected nodes <br> <h2>' + prev_topic + ' kewyords: ' + topic_keywords[p_id] +'</h2>'), 
                ('<h2>' + topic + ' keywords: ' + topic_keywords[ids[0]] + '</h2>'), 
                source_graph.data['top_papers_topic'][p_id], 
                source_graph.data['top_papers_topic'][ids[0]],
                source_graph.data['top_papers_6_10'][p_id],
                source_graph.data['top_papers_6_10'][ids[0]]
            );

        // if there is an edge between the two nodes, print the edge information
        } else {
            //find the edge id
            var edge_id;
            for (var i = 0; i<edge_list.length; i++){
                if (edge_list[i].includes(topic) && edge_list[i].includes(prev_topic)){
                    edge_id = i;
                }
            }
            console.log(edge_id);

            updateText(
                ('<h2>' + prev_topic + ' & ' + topic + ' kewyords: ' + edge_keywords[edge_id] +'</h2>'), 
                '', 
                top_papers_edge[edge_id], 
                '',
                edge_papers_6_10[edge_id],
                ''
            );
        }

        //display the trend data for both selected topics
        var prev_data_trend = source_topics_year.data[prev_topic];
        var data_trend= source_topics_year.data[topic];
        
        source_trend.data['prob'] = prev_data_trend;
        source_trend.data['comp_prob'] = data_trend;
        trend_legend.items[0].label.value = prev_topic;
        trend_legend.items[1].label.value = topic;
        plot_trend_line.visible = true;
        comp_trend_line.visible = true;
        source_trend.change.emit();

        //emit edge info to update colors
        source_edges.change.emit();
    }
    prev_id.data['prev'][0] = ids[0];
    prev_id.change.emit();

// if no nodes are selected, return the plot to normal
} else{ 
    //change the edge colors back to grey
    for (var i=1; i<source_edges.data['start'].length; i++){
        source_edges.data['color'][i] = '#444444';
    }
    source_edges.change.emit();
    
    //remove all graph related text
    updateText('','','','','','');

    //clear the trend plot
    plot_trend_line.visible = false;
    comp_trend_line.visible = false;
    trend_legend.items[0].label.value = "";
    trend_legend.items[1].label.value = "";

    //reset previous node id
    prev_id.data['prev'][0] = -1; 
    prev_id.change.emit();
}

"""
)

#when a node is selected, run the js_callback
source_graph.selected.js_on_change("indices", js_callback)

#add the graph_renderer to the plot
plot.renderers.append(graph_renderer)

#%%
#  Make the table
df_table = pd.DataFrame(index= G.nodes)

df_table['topic'] = df_table.index
topic_part = pd.Series(list(partition.values()), index= G.nodes)
df_table['partition'] = topic_part

df_topickeywords = pd.read_csv('data/topic_keywords.csv', index_col=0).loc[present_topics]
topic_keywords = df_topickeywords[df_topickeywords.columns[0:6]].apply(", ".join, axis= 1)[topic_part.index]
df_table['keywords'] = topic_keywords

probs = pd.Series([G.nodes[t]['size'] for t in G.nodes], index = G.nodes)
probs = probs/probs.max()
df_table['prob'] = probs

def downselect_prob(df):
    df = df.sort_values('prob', ascending=False).iloc[0:10]
    return df

df_table = df_table.groupby('partition').apply(downselect_prob)

fill_colors_table = [pal[i] for i in df_table['partition']]
df_table['fill_color'] = fill_colors_table
df_table.index = range(len(df_table))

from bokeh.models.widgets import DataTable, TableColumn, HTMLTemplateFormatter
from bokeh.plotting import figure, show
from bokeh.models.sources import ColumnDataSource

table_source = ColumnDataSource(df_table)

template="""                
            <div style="background:<%= 
                (function colorfromint(){
                    return(fill_color)
                    }()) %>;"> 
                <%= value %>
            </div>
            """

formatter =  HTMLTemplateFormatter(template=template)

columns = [
        TableColumn(field='partition', title='partition', formatter=formatter, width = 100),
        TableColumn(field='topic', title='Topic', width = 100),
        TableColumn(field='prob', title='Probability', width = 100),
        TableColumn(field='keywords', title='Keywords', width = 500),
          ]

data_table = DataTable(source=table_source,
                       columns=columns,
                       width = 800,
                       fit_columns=False
)



#%%
# make a plot with Louvian communities that combines the information of each topic in that community
num_years = len(df_topic_year.index)
# a list of the partition values [0,1,2,3]
communities = list(set(partition.values()))
# a list of the group names "group_1" etc
groups = []
for c in communities:
    groups.append('group_' + str(c))
group_year = df_topic_year.copy()
#create a df of group year filled with zeros
for g in groups:
    group_year[g] = group_year.iloc[:,-1] - group_year.iloc[:,-1]
group_year.drop(group_year.columns.difference(groups), 1, inplace=True)

#a function for getting the list of nodes in that group as well as their size and keywords in order of importance
def get_nodes(v, dict, GL):
    node_list = []
    node_size = 0
    # keyword strings is a list of empty strings the length of the number of topickeywords
    keywords_strings = ['']*df_topickeywords.shape[-1]
    for key, value in dict.items():
        # if the node is in the group of interest
        if v == value:
            # look through all the nodes in the new graph (GL)
            for n in GL.nodes:
                # add this node to the node_list and add the size to the group size
                if key == n:
                    node_list.append(n)
                    node_size += GL.nodes[n]['size']
                    # add the keywords to the string for this group in order of ranking
                    i=0
                    for word in df_topickeywords.loc[n]:
                        keywords_strings[i] += word +', '
                        i+=1
                    group_year[groups[v]] += df_topic_year[n]
    #average the group year
    group_year[groups[v]] = group_year[groups[v]] / len(node_list)
    return node_list, node_size, keywords_strings 

#https://networkx.org/documentation/networkx-1.10/reference/generated/networkx.algorithms.minors.contracted_nodes.html
GL = G.copy()
label_mapping = {}
color_mapping = {}
keywords = {}
#for each community, get the list of nodes in the community, the keywords, and the sum of the node size and then contract the nodes in the new graph and keep track of their color and keywords
for t in communities: 
    nodes_t, node_size, keywords_strings = get_nodes(t, partition, GL)
    u = nodes_t[0]
    for v in nodes_t[1:]:
        GL = contracted_nodes(GL, u, v, self_loops=False)
    GL.nodes[u]['size'] = node_size*.4
    for edge in GL.edges:
        GL.edges[edge]['weight'] = GL.edges[edge]['weight']*1.25
    label_mapping[u] = groups[t] 
    color_mapping[groups[t]] = pal[t]
    keywords[groups[t]] = keywords_strings


#change the indice names
GL = nx.relabel.relabel_nodes(GL, label_mapping)

#calculate the probability over the past 5 years
recent_group_prob = group_year.loc[2015:2020].mean()
recent_group_prob = recent_group_prob/recent_group_prob.max()
source_group_year = ColumnDataSource(data=group_year)

#calculate the colors based on partition (Louvian community) and recent probability
group_fill_colors = [color_mapping[n] for n in GL.nodes]
group_fill_colors = change_intensity(group_fill_colors, recent_group_prob)
group_keywords = [keywords[n] for n in GL.nodes]

#generate the plot
group_plot = figure(plot_width=1200, plot_height=850,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
#add a hover tool for nodes
group_node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("text", "@disp_text")])
group_plot.add_tools(group_node_hover_tool, TapTool())
group_plot.axis.visible = False

#render the graph
group_graph_renderer = from_networkx(GL, nx.kamada_kawai_layout, scale=0.8, center=(0, 0))
group_source_graph = group_graph_renderer.node_renderer.data_source
group_source_graph.add(group_fill_colors, 'color')
group_graph_renderer.node_renderer.glyph = Circle(radius = 'size', fill_color = 'color', radius_units='screen')

#plot line graph of group paper years and topics
g_years = group_year.index
group_source_trend = ColumnDataSource(data=dict(year=g_years, prob=[0]*len(g_years)))
group_plot_trend = figure(plot_width=700, plot_height=400, title = 'Group Probability Trend')

group_plot_trend.xaxis.axis_label = 'Year'
group_plot_trend.yaxis.axis_label = 'Normalized Topic Probability'

group_plot_trend_line = group_plot_trend.line(source=group_source_trend, x='year', y = 'prob', line_color='#D86EFF', line_width=2)
group_plot_trend_line.visible = False


group_words_list = Div(text= ''''<h3> Select a node to see the list of keywords for that community and a trend of topics in that community over time. </h3>''', width=600)

#define dictionary of variables to be used in callback
group_callback_args = dict(
    group_source_trend = group_source_trend,
    group_source_graph = group_source_graph,
    source_group_year=source_group_year,
    group_words_list =group_words_list,
    group_keywords=group_keywords,
    group_plot_trend_line =group_plot_trend_line,
 )

js_group_callback = CustomJS(args=group_callback_args, code = """

//get the id of the selected node
var ids = group_source_graph.selected.indices;

// if a node is selected, print the keywords and show the trendline
if (ids.length > 0){
    var group = group_source_graph.data['index'][ids[0]];
    console.log(group);

    group_words_list.text = '<h2>' + group + ' keywords: </h2> <h3>' + group_keywords[ids[0]] + '</h3>';
    group_words_list.change.emit();

    var group_data_trend = source_group_year.data[group];
    group_source_trend.data['prob'] = group_data_trend;
    group_plot_trend_line.visible = true;
    group_source_trend.change.emit();
        
}
// if no nodes are selected, hide the trend line and keywords
else{
    group_words_list.text = '<h3> Select a node to see the list of keywords for that community and a trend of topics in that community over time. </h3>'
    group_words_list.change.emit();
    group_plot_trend_line.visible = false
}

"""
)

#add the callback and graph renderer
group_source_graph.selected.js_on_change("indices", js_group_callback)
group_graph_renderer.edge_renderer.glyph = MultiLine(line_width = 'weight', line_alpha = 0.35, line_color="#444444")
group_plot.renderers.append(group_graph_renderer)

##%%
# make the layout of the page
layout =  column(
    intro_text,
    row(plot, column(topic_words_list, row(text_top_papers_topic, text_top_papers_6_10), column(topic_words_list_2, row(top_papers_topic_2, top_papers_6_10_2)))), 
    row(column(table_text, data_table), column(trend_text, plot_trend)),
    # group_text, 
    # row(group_plot, column(group_words_list, group_plot_trend))
    )

#%%
#save and show
output_fp = "output/ES_networkplot.html"

if not os.path.exists("output"): 
    os.mkdir('output')

output_file(filename=output_fp, title = 'Topic Modeling wtih LDA')

show(layout)

# %%
