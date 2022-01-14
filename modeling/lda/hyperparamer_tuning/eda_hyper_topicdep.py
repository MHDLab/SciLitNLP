#%%

import xyzpy
import numpy as np
import os

dataset_name = 'lda_hyper_20210127_topics.h5'
output_folder = r'C:\Users\aspit\Git\MHDLab-Projects\Energy Storage\topic_modeling\lda_hyperparamer_tuning\output'

ds = xyzpy.load_ds(os.path.join(output_folder,dataset_name))


ds = ds.squeeze()

ds
# %%


ds_sel = ds[['coherence_cv', 'coherence_umass']]

#https://github.com/RaRe-Technologies/gensim/issues/951#issuecomment-254022925
# ds_sel['perplexity'] = 2**(-ds_sel['perplexity'])



# ds_sel['perplexity'] = -np.log(ds_sel['perplexity'])

#%%





#%%
ds_sel.to_array('var').plot(row='var', sharey=False, col='eta', hue ='alpha_prefactor')
# %%



ds_sel['coherence_cv'].plot(row='eta', hue ='alpha_prefactor', xscale='log')
# %%
ds_sel['coherence_cv'].sel(alpha_prefactor=1).sel(num_topics = slice(50,150)).plot(hue='eta')
# %%
