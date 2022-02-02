#%%
import nlp_utils as nu
import pandas as pd
import json
import sqlite3
import os
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
from dotenv import load_dotenv
load_dotenv()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--regex', type=str, default = '', help="regular expression generated from indexed search (if using indexed search id generation)")
args = parser.parse_args()

fp_search_idx = os.path.join(os.getenv('REPO_DIR'), 'text_data/semantic/data/indexed_searches.json')
with open(fp_search_idx, 'r') as f:
    id_dict = json.load(f)

idxs = id_dict[args.regex]

db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)
cursor = con.cursor()
df = nu.fileio.load_df_semantic(con, idxs)
#%%
YEAR_RANGE = slice(1950,2019)

# %%
plt.figure(1)
year_counts_es = df['year'].value_counts().sort_index()
year_counts_es.loc[YEAR_RANGE].plot(marker='o')


#%%

year_counts = pd.read_csv('data/full_corpus_year_counts.csv',index_col=0)['year']
year_counts.index

#%%

plt.figure(2)
ratio = year_counts_es.divide(year_counts).dropna()
(ratio.loc[YEAR_RANGE]/1e-2).plot(marker='o')
# plt.ylim(0,1)
plt.xlabel('Year')

plt.suptitle('Abstract or Title Contains \'{}\''.format(args.regex))
plt.ylabel('% Annual Publications')
plt.savefig('output/search_term_pub_percent.png', facecolor='white')
# %%