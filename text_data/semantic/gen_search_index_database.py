#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu
import json
from dotenv import load_dotenv
load_dotenv()


db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
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

if not os.path.exists('data'): os.mkdir('data')
fp_search_idxs = 'data/indexed_searches.json'
if os.path.exists(fp_search_idxs):
    with open(fp_search_idxs, 'r') as f:
        id_dict = json.load(f)
else:
    id_dict = {}

regexes = ['%' + r + '%' for r in regexes]
for regex in regexes:

    print('Searching for regex: ' + regex)
    ids = nu.fileio.gen_ids_searchterm(
        con, 
        regex, 
        idx_name='id', 
        search_fields=['paperAbstract', 'title'], 
        search_limit=int(1e10), 
        output_limit=1e10
    )
    # all_ids.append(ids)

    id_dict[regex] = ids

with open(fp_search_idxs, 'w') as f:
    json.dump(id_dict, f, indent=2)
