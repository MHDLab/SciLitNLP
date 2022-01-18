#%%
import nlp_utils as nu
import pandas as pd
import json
import sqlite3
import os
import matplotlib.pyplot as plt

fp_search_idx = '../data/indexed_searches.json'
with open(fp_search_idx, 'r') as f:
    id_dict = json.load(f)

search_term = '%carbon nanotube%'
idxs = id_dict[search_term]
#%%


DATASET_DIR = r'E:'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

YEAR_RANGE = slice(1950,2019)
df = nu.fileio.load_df_semantic(con, idxs)
df.info()

# %%
year_counts_es = df['year'].value_counts().sort_index()
year_counts_es.loc[YEAR_RANGE].plot(marker='o')


#%%

year_counts = pd.read_csv('data/full_corpus_year_counts.csv',index_col=0)['year']
year_counts.index

#%%

ratio = year_counts_es.divide(year_counts).dropna()
(ratio.loc[YEAR_RANGE]/1e-2).plot(marker='o')
plt.ylim(0,1)
plt.xlabel('Year')

plt.suptitle('Abstract or Title Contains \'{}\''.format(search_term.strip('%')))
plt.ylabel('% Annual Publications')
plt.savefig('output/search_term_pub_percent.png', facecolor='white')
# %%