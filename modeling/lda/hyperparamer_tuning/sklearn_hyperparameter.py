#https://stackoverflow.com/questions/60613532/how-do-i-calculate-the-coherence-score-of-an-sklearn-lda-model

#%%

import pandas as pd
import pickle
import sqlite3
import os
import numpy as np
import sys

import xarray as xr
import xyzpy
import time

import git

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from tmtoolkit.topicmod.evaluate import metric_coherence_gensim


def main():

    repopath = git.Repo('.', search_parent_directories=True).working_tree_dir
    data_folder = os.path.join(repopath, 'data')

    con = sqlite3.connect(os.path.join(data_folder, 'seamsnlp.db'))
    cursor = con.cursor()

    df = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID')
    df_metadata = pd.read_sql_query("SELECT * FROM metadata", con, index_col='ID')

    df = pd.concat([df,df_metadata], axis=1).dropna(subset=['processed_text'])

    df = df[df['Session Name'] != 'Intro Materials'].dropna(subset=['Session Name'])

    df = df.sample(500, random_state=1)
    # %%



    text = df['processed_text'].values

    # maxx_features = 2**12
    vectorizer = CountVectorizer(max_features=None, min_df=5, max_df = 0.8, ngram_range=(1,1))
    X = vectorizer.fit_transform(text)

    #%%





    # 

    #%%


    def get_coherence(alpha, eta, num_topics):


        lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            doc_topic_prior=alpha,
            topic_word_prior=eta,
            max_iter=20, #is this the same as 'passes' in gensim?
            random_state=1
            )

        start = time.perf_counter()

        X_topics = lda_model.fit_transform(X)

        perplexity = lda_model.perplexity(X)

        coherence_umass = metric_coherence_gensim(measure='u_mass', 
                                top_n=25, 
                                topic_word_distrib=lda_model.components_, 
                                dtm=X, 
                                vocab=np.array([x for x in vectorizer.vocabulary_.keys()]), 
                                texts=text)

        coherence_umass = np.mean(coherence_umass)

        stop = time.perf_counter()
        ex_time = stop-start

        return perplexity, coherence_umass, ex_time


    runner = xyzpy.Runner(get_coherence, var_names=['perplexity', 'coherence_umass', 'ex_time'])

    #https://stats.stackexchange.com/questions/349761/reasonable-hyperparameter-range-for-latent-dirichlet-allocation

    combos = {
        'num_topics': np.logspace(1,2.5,10).astype(int),
        'alpha': [0.01, 0.02, 0.05, 0.1, 0.2, 1],
        'eta': [0.01, 0.02, 0.05, 0.1, 0.2, 1]
        }



    ds = runner.run_combos(combos)


    #%%

    ds = ds.squeeze()
    ds.to_dataframe().to_csv(r'C:\Users\aspitarl\Git\MHDLab-Projects\NLP_MHD\analysis\lda_hyperparamer_tuning\lda_hyper_sklearn.csv')




if __name__ == '__main__':
    main()

#%%








