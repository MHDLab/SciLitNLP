#%%


import gensim
# from gensim.models import LdaModel

import pandas as pd
import sqlite3
import os
import numpy as np
import sys

import xarray as xr
import numpy as np
import xyzpy

import time

sys.path.append(r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\nlp_utils')

from nlp_utils import gensim_utils, sklearn_utils

def main():

    dataset_name = 'texts_bigram_100.h5'
    output_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\output'

    ds_texts = xyzpy.load_ds(os.path.join(output_folder,dataset_name))

    # ds_texts = ds_texts.isel(ID=slice(0,100))

    IDs = ds_texts.coords['ID'].values

    def get_coherence(data_vars, threshold, min_count, alpha_prefactor,**lda_kwargs):

        texts_bigram = ds_texts['texts_bigram'].sel(threshold=threshold, min_count=min_count).values

        texts_bigram = [t.split() for t in texts_bigram]

        lda_kwargs['alpha'] = alpha_prefactor/lda_kwargs['num_topics']


        id2word = gensim.corpora.Dictionary(texts_bigram)
        data_words = [id2word.doc2bow(doc) for doc in texts_bigram]



        start = time.perf_counter()

        lda_model = gensim.models.LdaModel(
                                        data_words,
                                        id2word=id2word,
                                        random_state=42,
                                        **lda_kwargs
        )

        metrics = []

        if 'perplexity' in data_vars:
            perplexity = lda_model.log_perplexity(data_words)
            metrics.append(perplexity)

        if 'coherence_cv' in data_vars:
            coherence_model = gensim.models.CoherenceModel(model=lda_model, texts=texts_bigram, dictionary=id2word, coherence='c_v', processes=1)
            coherence_cv = coherence_model.get_coherence()
            metrics.append(coherence_cv)

        if 'coherence_umass' in data_vars:
            coherence_model = gensim.models.CoherenceModel(model=lda_model, texts=texts_bigram, dictionary=id2word, coherence='u_mass', processes=1)
            coherence_umass = coherence_model.get_coherence()
            metrics.append(coherence_umass)

        if 'ex_time' in data_vars:
            stop = time.perf_counter()
            ex_time = stop-start
            metrics.append(ex_time)
        
        if 'bigram_fraction' in data_vars:
            num_bigrams, total_words = gensim_utils.bigram_stats(data_words, id2word)
            bigram_fraction = num_bigrams/total_words
            metrics.append(bigram_fraction)

        return_val = tuple(metrics)

        df_topickeywords, doc_topic_probs = gensim_utils.gensim_topic_info(lda_model, data_words, id2word)
        df_topickeywords = df_topickeywords.apply("-".join, axis=1)

        df_doc_topic_probs = pd.DataFrame(doc_topic_probs)
        df_doc_topic_probs.index = IDs

        tsne_x, tsne_y = sklearn_utils.calc_tsne(texts_bigram)
        return_val = return_val + tuple([tsne_x, tsne_y, df_topickeywords.values, doc_topic_probs])

        return return_val

    var_names=[
    'perplexity', 
    'coherence_cv',
    'coherence_umass',
    # 'ex_time' #Gives inconsistent results...
    'bigram_fraction',
    # 'tsne_x',
    # 'tsne_y',
    # 'topic_keywords',
    # 'doc_topic_probs'
    ]

    var_dims = {
        # 'tsne_x': ['ID'],
        # 'tsne_y': ['ID'],
        # 'topic_keywords': ['topic'],
        # 'doc_topic_probs': ['ID', 'topic']
        }

    var_coords = {
        # 'ID': IDs,
        # 'topic': [i+1 for i in list(range(20))]
    }

    runner = xyzpy.Runner(get_coherence, var_names=var_names, var_dims=var_dims, var_coords=var_coords, constants={'data_vars': var_names})


    output_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\output'

    h = xyzpy.Harvester(runner, data_name=os.path.join(output_folder,'lda_hyper_20210127_topics.h5'))


    #https://stats.stackexchange.com/questions/349761/reasonable-hyperparameter-range-for-latent-dirichlet-allocation


    combos = {
        'passes': [3],
        'num_topics': [int(n) for n in np.logspace(1.31,2.699,20)],
        # 'alpha': np.logspace(-2,0,5),
        'alpha_prefactor': np.logspace(0,2,3),
        'eta': np.logspace(-2.5,-.5,3),
        'threshold': [100],
        'min_count': [10]
        }


    #The data should not conflict, But overwrite just in case
    h.harvest_combos(combos, parallel=True, overwrite=True)

    # ds = runner.run_combos(combos, parallel=True, constants={'data_vars': var_names})

    #%%

    # ds = ds.squeeze()
    # ds.to_dataframe().to_csv(r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\lda_hyper_bigram.csv')


if __name__ == '__main__':
    main()
