"""
Exploratory data analysis of the semantic scholar dataset. 
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from nlp_utils import gensim_utils, sklearn_utils, fileio
data_folder = r'C:\Users\byahn\Code\MLEF\ES_TextData\data'
df = fileio.load_df(os.path.join(data_folder, 'SOC_ES.db'))

fig_path = r'C:\Users\byahn\Code\MLEF\MLEF-Energy-Storage.github.io'

def citations_to_list(text):
    text = text.strip("][").split(', ')
    text = [t.replace("'","") for t in text]
    return text

df['inCitations'] = df['inCitations'].apply(citations_to_list)
df['outCitations'] = df['outCitations'].apply(citations_to_list)

#%%

df.head()
# %%
fos = df['fieldsOfStudy'].str.strip('][').str.split(', ').apply(lambda x: x[0])
barlist = fos.value_counts(ascending=False).plot(kind='bar', figsize=(15,5), title="Distribution of Abstract Field of Study", ylabel="Number of Abstracts (54,125 total)", color=['#6DA367']*len(fos.value_counts()), fontsize=15)
plt.tight_layout
plt.savefig(os.path.join(fig_path,'FOS.png'), bbox_inches="tight")

#fos.hist(figsize=(15,5))
#plt.xticks(rotation=45)
# %%

df['Year'].dropna().astype(int).value_counts().sort_index(ascending=False).plot(marker='o', figsize=(15,5), title="Years of Abstract Publications", ylabel="Number of Abstracts (54,125 total)", color=['#6DA367']*len(df['Year'].dropna().astype(int).value_counts()), fontsize=15)
#df['Year'].groupby()
plt.savefig(os.path.join(fig_path,'Years.png'), bbox_inches="tight")

# %%
df['raw_text'].dropna().apply(len).hist(bins = 100, figsize=(5,5), color="#6DA367")
plt.title('Distribution of Abstract Length')
plt.ylabel('Number of Abstracts')
plt.xlabel('Length of Abstract (characters)')
plt.xlim(0,8000)
plt.xticks(np.arange(0,8000, step=1000))
plt.savefig(os.path.join(fig_path,'Length.png'), bbox_inches="tight")

#%%
bins = np.logspace(np.log10(0.9),np.log10(20000), 100)
df['inCitations'].apply(len).value_counts().hist(bins=bins, figsize=(5,5), color="#6DA367")

plt.title('Distribution of Paper Citations')
plt.ylabel('Number of Papers')
plt.xlabel('Number of citations in literature')
plt.yscale('log')
plt.xscale('log')
plt.savefig(os.path.join(fig_path,'InCite.png'), bbox_inches="tight")

# %%
bins = np.logspace(np.log10(0.9),np.log10(20000), 100)
df['outCitations'].apply(len).value_counts().hist(bins=bins,figsize=(5,5), color="#6DA367")

plt.title("Distribution of References")
plt.ylabel('Number of Papers')
plt.xlabel('Number of Citations in each Paper')
plt.yscale('log')
plt.xscale('log')
plt.savefig(os.path.join(fig_path,'OutCite.png'), bbox_inches="tight")
# %%
