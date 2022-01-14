#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})

DATASET_DIR = r'E:'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

YEAR_RANGE = slice(1950,2019)
#%%

years = nu.fileio.get_columns_as_df(con, ['year'])['year'].astype(int)
years
#%%

year_counts = years.value_counts().sort_index()
(year_counts.loc[YEAR_RANGE]/1e6).plot(marker='o')
plt.xlabel('Year')

plt.suptitle('Full Semantic Scholar Database')
plt.ylabel('Annual Publications (Millions)')

# %%

idxs = pd.read_csv('data/indexed_searches.csv')['%energy storage%']
idxs

df = nu.fileio.load_df_semantic(con, idxs.values)
df.info()

# %%
year_counts_es = df['year'].value_counts().sort_index()
year_counts_es.loc[YEAR_RANGE].plot(marker='o')


#%%
year_counts.index

#%%

ratio = year_counts_es.divide(year_counts).dropna()
(ratio.loc[YEAR_RANGE]/1e-2).plot(marker='o')
plt.ylim(0,1)
plt.xlabel('Year')

plt.suptitle('Abstract or Title Contains \'Energy Storage\'')
plt.ylabel('% Annual Publications')
# %%
