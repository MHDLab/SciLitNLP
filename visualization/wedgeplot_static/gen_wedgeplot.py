"""
Generate the wedge plot html file. 
"""

#%%
import pandas as pd
import os
import sqlite3
from nlp_utils import fileio

#The texts that were used in gen_data.py must be used, this data is used for plot hover data. Perhaps should just be included in df_metadata. 

# db_folder = r'E:\\'
# con = sqlite3.connect(os.path.join(db_folder, 'soc.db'))
# regex = '%energy storage%'
# ids = fileio.gen_ids_regex(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e6), output_limit=int(3e4))
# df = fileio.load_df_semantic(con, ids)

db_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs'
df = fileio.load_df_SEAMs(db_folder).dropna(subset=['processed_text'])
# df = df.sample(300, random_state=42)


BASE_DIR =  os.path.dirname(os.path.abspath(__file__))
df_meta = pd.read_csv(os.path.join(BASE_DIR,'data','df_meta.csv'), index_col=0)
df_doc_topic_probs = pd.read_csv(os.path.join(BASE_DIR,'data','df_doc_topic_probs.csv'), index_col=0)
df_wedges = pd.read_csv(os.path.join(BASE_DIR,'data','df_wedges.csv'), index_col=[0,1])

df_topickeywords = pd.read_csv(os.path.join(BASE_DIR,'data','df_topickeywords.csv'), index_col=0)
topic_strs = df_topickeywords[df_topickeywords.columns[0:5]].apply("_".join, axis=1)


df = pd.concat([df_meta, df[['display_url', 'title', 'year']]],axis=1)

# %%

#Able to just have the plot show a single color for the top topic. This was the original plot essentially from MLEF 2020. This reduces computational resources. 
#TODO: make this a slider somehow, I got close with global variables
# PLOT_MODE = 'circle'
PLOT_MODE = 'wedge'

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
output_notebook()

from bokeh.palettes import Category20
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, OpenURL, HoverTool, CustomJS, Slider, TapTool, TextInput, Div, Paragraph, Legend, RangeSlider

colors = Category20[20]

#Topics ordered by total cumulative probability for better color scheme
topics_ordered = df_doc_topic_probs.sum().sort_values(ascending=False).index

hover_cols = ['title', 'year', 'top_topics_str']
tooltips = []
for col in hover_cols:
        tooltips.append((col, "@" + col + "{safe}"))

x_buff = (df['tsne_x'].max() - df['tsne_x'].min())/100
x_range = (df['tsne_x'].min()- x_buff, df['tsne_x'].max() + x_buff)
y_buff = (df['tsne_y'].max() - df['tsne_y'].min())/100
y_range = (df['tsne_y'].min()- y_buff, df['tsne_y'].max() + y_buff)

p = figure(tools=[HoverTool(tooltips=tooltips, point_policy="follow_mouse" ,names = ['hover_plot']), 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap', 'crosshair'], output_backend="webgl", plot_width=1500, plot_height=700, x_range = x_range, y_range = y_range)

source_hover = ColumnDataSource(data=dict(tsne_x=[], tsne_y=[], url = [], title=[], Year =[], top_topics_str =[], display_text=[]))

def update_hover_data(id_subset= None):
    if id_subset is not None:
        ids = [id for id in df.index if id in id_subset]
        source_hover.data = df.loc[ids]
    else:
        source_hover.data = df

update_hover_data()

source_wedges = {'topic_' + str(i):
    ColumnDataSource(
        data = dict(x = [], y = [], start_angle= [], end_angle = [])
        ) for i in range(1,21)
}

def update_wedge_data(topic, id_subset = None):
    if PLOT_MODE == 'wedge':
        df_topic = df_wedges.loc[topic]
    else:# PLOT_MODE == 'circle':
        df_topic = df[df['top_topic']==topic]

    if id_subset is not None:
        ids = [id for id in df_topic.index if id in id_subset]
        df_topic = df_topic.loc[ids]

    if PLOT_MODE == 'wedge':
        #Gives error for circle...TODO: figure out
        df_topic['tsne_x'] = df['tsne_x'].loc[df.index.intersection(df_topic.index)]
        df_topic['tsne_y']= df['tsne_y'].loc[df.index.intersection(df_topic.index)]

    source_wedges[topic].data = df_topic

legend_items = []
wedge_renderers = []
num_topics = len(df_doc_topic_probs.columns)

radius_units = "data"
glyph_radius = 0.5

# radius_units = "screen"
# glyph_radius = 5
#A renderer is generated for each topic (i.e. circle or wedge fragment.)

for i, topic in enumerate(topics_ordered):
    if PLOT_MODE == 'wedge':
        r = p.wedge(source = source_wedges[topic], color=colors[i], alpha=1, direction="clock", radius=glyph_radius, radius_units=radius_units, x='tsne_x', y='tsne_y',  start_angle='slice_start', end_angle='slice_end',)
    else:
        r = p.circle(source=source_wedges[topic], color=colors[i], alpha=1,radius=glyph_radius, radius_units=radius_units, x='tsne_x', y='tsne_y')

    wedge_renderers.append(r)

    legend_str = str(topic) + ': ' + topic_strs[topic]
    legend_items.append((legend_str , [r]))

    update_wedge_data(topic)

hover_renderer = p.circle('tsne_x', 'tsne_y', name = 'hover_plot', source=source_hover, radius=glyph_radius, alpha=0, radius_units=radius_units)

legend1 = Legend(items=legend_items,  orientation="vertical", click_policy="hide", title='Topics (click to hide/show)')
p.add_layout(legend1, 'left')


summary_text_description = Div(text = '<h2> Click on a point to display a link to the paper. (Use zoom tool for larger markers)</h2><br>')
summary_text = Div(text='')
layout = row(p, column([summary_text_description,summary_text]))

# handle the currently selected article
def selected_code():
    code = """
            var texts = [];
            cb_data.source.selected.indices.forEach(index => texts.push(source.data['display_text'][index].slice(0,2000)));

            var text = "<h4>" + texts[0].toString() + "</h4>";

            current_selection.text =  text
            current_selection.change.emit();
        """

    return code

callback_selected = CustomJS(args=dict(source=source_hover, current_selection=summary_text), code=selected_code())
taptool = p.select(type=TapTool)
taptool.callback = callback_selected

from bokeh.resources import CDN
from bokeh.embed import file_html, components, json_item
import json 

html = file_html(layout, CDN, "wedge plot")

with open(r'wedgeplot.html', 'w') as file:
    file.write(html)
# %%
