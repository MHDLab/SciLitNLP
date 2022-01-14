#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu

DATASET_DIR = r'E:'
db_path = os.path.join(DATASET_DIR, 'soc.db')
con = sqlite3.connect(db_path)

#TODO: I wonder if it might be faster doing first pass to see if there were any of the search terms, then search for each term in that subset.
regexes = [
    'energy storage',
    # 'carbon nanotube',
    # 'electricity storage',
    # 'lithium ion',
    # 'lead acid',
    # 'solid oxide fuel cell',
    # 'compressed air',
    # 'pumped thermal',
    # 'thermomechanical',
    # 'thermal energy storage', #just thermal would probably be a lot of articles...
    # 'flywheel',
    # 'superconducting magnetic',
    # 'supercapacitor',
]

regexes = ['%' + r + '%' for r in regexes]

all_ids = []
for regex in regexes:

    print('Searching for regex: ' + regex)
    ids = nu.fileio.gen_ids_searchterm(con, regex, idx_name='id', search_fields=['paperAbstract', 'title'], search_limit=int(1e10), output_limit=1e10)
    all_ids.append(ids)


df = pd.DataFrame(all_ids).transpose()
df.columns = regexes

fp_search_idxs = 'data/indexed_searches.csv'


df.to_csv(fp_search_idxs)