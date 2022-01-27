#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from dotenv import load_dotenv
load_dotenv()


db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

YEAR_RANGE = slice(1950,2019)
#%%

years = nu.fileio.get_columns_as_df(con, ['year'])['year'].astype(int)
years
#%%
if not os.path.exists('data'): os.mkdir('data')
year_counts = years.value_counts().sort_index()
year_counts.to_csv('data/full_corpus_year_counts.csv')

#%%

(year_counts.loc[YEAR_RANGE]/1e6).plot(marker='o')
plt.xlabel('Year')

plt.suptitle('Full Semantic Scholar Database')
plt.ylabel('Annual Publications (Millions)')

# plt.tight_layout()
plt.savefig('output/full_corpus_pub_annual.png', facecolor='white')
# %%
