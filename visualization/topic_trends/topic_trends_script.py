s_topic_words = nu.corex_utils.get_s_topic_words(topic_model, 10)
df_doc_topic_probs = pd.DataFrame(topic_model.p_y_given_x, index=df_tm.index , columns=s_topic_words.index)
df_topicsyear = nu.common.calc_topics_year(df_doc_topic_probs, df_tm['year'], norm_each_topic=False)


from importlib import reload
reload(nu.plot)

highlight_topics = ['topic_' + str(i) for i in range(len(corex_anchors))]

year_range_fit = slice(2015,2020)
year_range_plot = slice(1990,2020)

nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, highlight_topics=highlight_topics)

plt.savefig('output/top_slopes_plot.png')


nu.plot.top_slopes_plot(df_topicsyear.loc[year_range_plot], s_topic_words, year_range_fit, n_plots=10, highlight_topics=highlight_topics,ascending=True)

plt.savefig('output/neg_slopes_plot.png')

