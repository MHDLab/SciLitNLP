#%%

import xyzpy
import os

dataset_name = 'lda_hyper_20210126.h5'
output_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\output'

ds = xyzpy.load_ds(os.path.join(output_folder,dataset_name))


ds = ds.squeeze()

ds
# %%


ds = ds[['coherence_cv', 'coherence_umass', 'perplexity','bigram_fraction']]

#%%

ds = ds.drop(1,'eta')


# %%
ds['coherence_cv'].plot(row='min_count',col='threshold', hue = 'eta')


# %%
ds['bigram_fraction'].plot(row='min_count',col='threshold', hue = 'eta')
# %%
