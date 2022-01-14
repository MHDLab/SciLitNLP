import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import xarray as xr

import json

fp = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\topic_modeling\lda_hyperparamer_tuning\output\lda_hyper_full.h5'
ds_lda = xr.load_dataset(fp)
ds_lda = ds_lda.squeeze()

ds_lda.coords['alpha'] = [np.log10(alpha) for alpha in ds_lda.coords['alpha'].values]
ds_lda.coords['eta'] = [np.log10(eta) for eta in ds_lda.coords['eta'].values]

import sqlite3
fp_rawdata = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\data\nlp_justenergystorage.db'
con = sqlite3.connect(fp_rawdata )
cursor = con.cursor()
df_text = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID').dropna()

df_text = df_text.loc[ds_lda.coords['ID']]

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

def gen_slider_coord(coords, id):
    slider = dcc.Slider(
            id=id,
            min=coords.min().item(),
            max=coords.max().item(),
            value = coords.min().item(),
            marks = {str(val): str(val) for val in coords.values},
            step=None
        )

    return slider

controls = html.Div([

        html.Label("Min Bigram Count"),
        gen_slider_coord(ds_lda.coords['min_count'], 'min_count-slider'),

        html.Label("Log(Alpha)"), 
        gen_slider_coord(ds_lda.coords['alpha'], 'alpha-slider'),

        html.Label("Bigram Threshold"),
        gen_slider_coord(ds_lda.coords['threshold'], 'bigram_threshold-slider'),

        html.Label("Log(Eta)"), 
        gen_slider_coord(ds_lda.coords['eta'], 'eta-slider'),
    ],
    style = {'columnCount': 2})

app.layout = html.Div([
    controls,
    html.Div([
        html.Div(children=[
            html.H4('Tsne plot'),
            dcc.Graph(id='tsne-graph'),

        ],
        className='six columns'),
        html.Div(children=[
            html.H4('Coherence CV score'),
            dcc.Graph(id='heatmap-cv')
        ] ,className='six columns')

        ],
         className='row'
    ),
    html.P(id='paper-description'),
    html.Div(dcc.Tab(id='keyword-table'), className='six columns'),
    ]
)

colorscale = px.colors.qualitative.Light24


@app.callback(
    Output('tsne-graph', 'figure'),
    Input('alpha-slider', 'value'),
    Input('min_count-slider', 'value'),
    Input('eta-slider', 'value'),
    Input('bigram_threshold-slider', 'value')
)
def tsne_plot(alpha, min_count, eta, bigram_threshold):

    filtered_ds = ds_lda.sel(alpha=alpha, min_count=min_count, eta=eta, threshold=bigram_threshold)
    top_topics = filtered_ds['doc_topic_probs'].argmax('topic').values
    top_topics = [str(top) for top in top_topics]

    df = filtered_ds[['tsne_x', 'tsne_y']].to_dataframe()
    df['top_topic'] = top_topics
    df['title'] = df_text['title']
    df['url'] = df_text['url']
    df['raw_text'] = df_text['raw_text']
    
    color_map = {str(i): colorscale[i] for i in range(len(filtered_ds.coords['topic']))}
    hover_data = {
        'title': True,
        'tsne_x': False,
        'tsne_y': False
    }

    fig = px.scatter(df, x='tsne_x', y='tsne_y', color='top_topic', width = 1000, height=600, color_discrete_map=color_map, hover_data=hover_data, custom_data=['title', 'url', 'raw_text'])
    fig.update_layout(transition_duration=500)

    return fig


@app.callback(
    Output('paper-description', 'children'),
    Input('tsne-graph', 'clickData'))
def display_text_callback(clickData):

    if clickData is not None:
        custom_data = clickData['points'][0]['customdata']
        # json.dumps(clickData, indent=2)
        # out_str = '<a href=' + custom_data[0] + '>' + 'Abstract: ' + custom_data[1] + '</a>'
        out = html.Div([
                html.A(custom_data[0], href=custom_data[1]),
                html.P('Abstract Text: ' + custom_data[2])
            ])
        return out


@app.callback(
    Output('heatmap-cv', 'figure'),
    Input('alpha-slider', 'value'),
    Input('min_count-slider', 'value'),
    Input('eta-slider', 'value'),
    Input('bigram_threshold-slider', 'value')
)
def metric_plot(alpha, min_count, eta, bigram_threshold):
    fig = make_subplots(2)

    #Alpha eta plot
    filtered_ds = ds_lda.sel(min_count=min_count, threshold=bigram_threshold)

    data = filtered_ds['coherence_cv'].values
    alphas = filtered_ds.coords['alpha'].values
    etas = filtered_ds.coords['eta'].values

    colorplot_alphaeta = go.Heatmap(z=data, x=alphas , y=etas,colorbar_y=0.75, colorbar_len=0.5)
    marker_alphaeta = go.Scatter(x=[alpha], y=[eta], marker_color='black', marker_symbol='x', marker_size=20)

    fig.add_trace(colorplot_alphaeta)
    fig.add_trace(marker_alphaeta)    

    fig.update_xaxes(row=1,col=1, title_text='log(Alpha)')
    fig.update_yaxes(row=1,col=1, title_text='log(Eta)')

    #Bigram plot
    filtered_ds = ds_lda.sel(alpha=alpha, eta=eta)

    data = filtered_ds['coherence_cv'].values
    min_counts = filtered_ds.coords['min_count'].values
    thresholds = filtered_ds.coords['threshold'].values

    colorplot_bigram = go.Heatmap(z=data, x=min_counts, y=thresholds,colorbar_y=0.25, colorbar_len=0.5)
    marker_bigram = go.Scatter(x=[min_count], y=[bigram_threshold], marker_color='black', marker_symbol='x', marker_size=20)

    fig.add_trace(colorplot_bigram, row=2, col=1)
    fig.add_trace(marker_bigram, row=2, col=1)    

    fig.update_xaxes(row=2,col=1, title_text='Min bigram count')
    fig.update_yaxes(row=2,col=1, title_text='Bigram threshold')

    fig.update_layout(transition_duration=500, width = 800, height=600, showlegend=False)

    return fig


def generate_table(dataframe,topic_probs):
    return html.Table([
        html.Thead(
            html.Tr([html.Th('Topic ' + str(col-1) + '   (' + '{:.5f}'.format(topic_probs[col-1]) + ')',style={'color' : colorscale[col-1]}) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(len(dataframe))
        ])
    ])

@app.callback(
    Output('keyword-table', 'children'),
    Input('alpha-slider', 'value'),
    Input('min_count-slider', 'value'),
    Input('eta-slider', 'value'),
    Input('bigram_threshold-slider', 'value')
)
def topic_keyword_table(alpha, min_count, eta, bigram_threshold):
    filtered_ds = ds_lda.sel(alpha=alpha, min_count=min_count, eta=eta, threshold=bigram_threshold)
    s = filtered_ds['topic_keywords'].to_series()
    s= s.str.split('-')
    df = pd.DataFrame.from_dict(dict(zip(s.index, s.values)))
    
    topic_probs = filtered_ds['doc_topic_probs'].mean('ID')
    topic_probs = topic_probs/topic_probs.sum('topic')
    top_topics = topic_probs.sortby(topic_probs, ascending=False).coords['topic'].values

    df = df[top_topics]
    tab = generate_table(df,topic_probs.values)
    return tab



if __name__ == '__main__':
    app.run_server(debug=True)    