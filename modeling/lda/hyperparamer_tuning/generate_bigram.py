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

    data_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\data'

    con = sqlite3.connect(os.path.join(data_folder, 'nlp_justenergystorage_100.db'))
    cursor = con.cursor()

    df = pd.read_sql_query("SELECT * FROM processed_text", con, index_col='ID').dropna(subset=['processed_text'])
    df = df[df['language'] == 'en']

    # df = df.sample(500)

    #%%
    texts = df['processed_text'].values
    texts = [t.split() for t in texts]

    def gen_bigram(data_vars, threshold, min_count):

        bigram = gensim.models.Phrases(texts, min_count=min_count, threshold=threshold)
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        texts_bigram = [bigram_mod[doc] for doc in texts]

        

        id2word = gensim.corpora.Dictionary(texts_bigram)
        data_words = [id2word.doc2bow(doc) for doc in texts_bigram]

        num_bigrams, total_words = gensim_utils.bigram_stats(data_words, id2word)

        bigram_fraction = num_bigrams/total_words

        texts_bigram = [" ".join(l) for l in texts_bigram]

        return texts_bigram, bigram_fraction



    var_names=[
    'texts_bigram',
    'bigram_fraction',
    ]

    var_dims = {
        'texts_bigram': ['ID'],
        }

    var_coords = {
        'ID': df.index.values,
        # 'topic': [i+1 for i in list(range(20))]
    }

    runner = xyzpy.Runner(gen_bigram, var_names=var_names, var_dims=var_dims, var_coords=var_coords, constants={'data_vars': var_names})


    output_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\output'

    h = xyzpy.Harvester(runner, data_name=os.path.join(output_folder,'texts_bigram_100.h5'))


    #https://stats.stackexchange.com/questions/349761/reasonable-hyperparameter-range-for-latent-dirichlet-allocation


    combos = {
        'threshold': np.logspace(0,3,4),
        'min_count': np.logspace(0,2,3)
        }


    #The data should not conflict, But overwrite just in case
    h.harvest_combos(combos, parallel=True, overwrite=True)

    # ds = runner.run_combos(combos, parallel=True, constants={'data_vars': var_names})

    #%%

    # ds = ds.squeeze()
    # ds.to_dataframe().to_csv(r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\lda_hyper_bigram.csv')


if __name__ == '__main__':
    main()
