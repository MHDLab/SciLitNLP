
#%%
import matplotlib.pyplot as plt
import pandas as pd
from nlp_utils import fileio

db_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs'
df = fileio.load_df_SEAMs(db_folder).dropna(subset=['processed_text'])

#%%

seam_counts = df['SEAM'].value_counts().sort_index()
seam_counts.name = 'num articles'
seam_counts.plot(legend=True)

#%%

number_misspelled = [len(s.split(" ")) for s in df['misspelled']]

df['number_misspelled'] = number_misspelled

df['number_misspelled'].hist()

#%% 

misspelled = df[['number_misspelled', 'SEAM']].groupby('SEAM').mean()
misspelled.name = 'avg_mispelled'
misspelled.plot()

#%%




#%%

df['text_length'] = df['OCR_text'].apply(len)

avg_text_length = df[['text_length', 'SEAM']].groupby('SEAM').mean()

avg_text_length.plot()

#%%



df['frac_misspelled'] = df['number_misspelled']/df['text_length']

avg_frac_misspelled = df[['frac_misspelled', 'SEAM']].groupby('SEAM').mean()

avg_frac_misspelled.plot()

#%%

cutoff_percentage = 0.05

df_cut =df.where(df['frac_misspelled'] > cutoff_percentage )

df_cut = df_cut.dropna(how='all')

print(str(len(df_cut)), 'papers have greater than', cutoff_percentage*100, '% misspelled words')

df['frac_misspelled'].plot.hist()

plt.vlines(cutoff_percentage,*plt.gca().get_ylim(), color='red')

#%%
print('Most misspelled dataframes')
df.sort_values(by='frac_misspelled', ascending=False)[['title', 'SEAM', 'Session Name', 'frac_misspelled']].head()


# %%
