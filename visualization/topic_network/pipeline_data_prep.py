import pandas as pd
import os
import nlp_utils.common as nu_common 

def pipeline_data_prep(df, df_topickeywords, df_doc_topic_probs, df_doc_edge_probs, da_sigma, min_edge_weight):
    s_year = pd.Series(df['year'].dropna(), index=df.index)


    df_topicsyear = nu_common.calc_topics_year(df_doc_topic_probs, s_year, norm_each_topic=False)
    df_topicsyear.to_csv(os.path.join('data','topic_year.csv'))


    top_papers_topic = pd.Series(index= df_doc_topic_probs.columns)
    top_topic_paper = df_doc_topic_probs.idxmax(axis=1)
    top_papers_6_10 = pd.Series(index= df_doc_topic_probs.columns)

    for topic in df_doc_topic_probs.columns:
    #first part of string
        top_papers = df_doc_topic_probs[topic].sort_values(ascending=False)
        text = '<h3>Papers with highest probability for selected topic: </h3>'

        for idx in top_papers[0:5].index:
            prob = top_papers[idx]
        # idx = str(idx)

            linkstr = df['title'][idx] + " (" + str(df['year'][idx]) + ")"
            text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
            text += " (topic prob: {:0.1f}%)".format(prob*100)
        # text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
            text += " <br><br> "
    
        top_papers_topic[topic] = text


    #
        papers = top_topic_paper[top_topic_paper == topic].index
    # log_probs = df.loc[papers]['logprob'].sort_values(ascending=False)[0:5]
        text = '<h3> </h3>'

        for idx in top_papers[6:11].index:
            prob = top_papers[idx]

            linkstr = df['title'][idx] + " (" + str(df['year'][idx]) + ")"
            text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
            text += " (topic prob: {:0.1f}%)".format(prob*100)
        # text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
            text += " <br><br> "

    # top_papers_str = ", ".join(df['title'][top_papers.index])
    
        top_papers_6_10[topic] = text


    top_papers_topic.to_csv(os.path.join('data','top_papers_topic.csv'))
    top_papers_6_10.to_csv(os.path.join('data','top_papers_6_10.csv'))


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

            linkstr = df['title'][idx] + " (" + str(df['year'][idx]) + ")"
            text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
            text += " (topic prob: {:0.1f}%)".format(prob*100)
        # text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
            text += "<br><br> "
    
        top_papers_edge[edge] = text

    #
        papers = top_edge_paper[top_edge_paper == edge].index
    # log_probs = df.loc[papers]['logprob'].sort_values(ascending=False)[0:5]
        text = '<h3> </h3>'

        for idx in top_papers[6:10].index:
            prob = top_papers[idx]
        
            linkstr = df['title'][idx] + " (" + str(df['year'][idx]) + ")"
            text += " <a href=" + df['display_url'][idx] + ">" + linkstr + "</a><br>"
            text += " (topic prob: {:0.1f}%)".format(prob*100)
        # text += " (# citations {:})".format(len(df['inCitations'][idx].split(',')))
            text += "<br><br> "

    # top_papers_str = ", ".join(df['title'][top_papers.index])
    
        edge_papers_6_10[edge] = text
    
    top_papers_edge.to_csv(os.path.join('data','top_papers_edge.csv'))
    edge_papers_6_10.to_csv(os.path.join('data','edge_papers_6_10.csv'))



    print('Generating Graph')

    import networkx as nx 
    G = nx.Graph() 

    for topic_i in da_sigma.coords['topic_i'].values:
        for topic_j in da_sigma.coords['topic_j'].values:
            weight = da_sigma.sel(topic_i=topic_i, topic_j=topic_j).item()
            if  weight > min_edge_weight:  
                if topic_i != topic_j:
                    G.add_edge(topic_i,topic_j, weight=weight)

    topic_keywords = df_topickeywords[df_topickeywords.columns[0:6]].apply(", ".join, axis= 1)

    for node in G.nodes:
        G.nodes[node]['disp_text'] = topic_keywords[node].replace(',', '\n')
        G.nodes[node]['size'] = df_doc_topic_probs.sum()[node]


    print('Writing graph to disk')

    nx.write_gexf(G, os.path.join('data','G_topic.gexf'))