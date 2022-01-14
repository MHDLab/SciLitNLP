#%%

import xyzpy
import os

dataset_name = 'texts_bigram_100.h5'
output_folder = r'C:\Users\aspitarl\Git\MHDLab-Projects\Energy-Storage\topic_modeling\lda_hyperparamer_tuning\output'

ds = xyzpy.load_ds(os.path.join(output_folder,dataset_name))

ds = ds.squeeze()

ds
# %%
ds['bigram_fraction'].plot(hue='min_count')
# %%
ds['bigram_fraction'].to_dataframe().sort_values('bigram_fraction')
# %%
