#%%

import sqlite3
import os
import pandas as pd
import nlp_utils as nu
import json
import argparse
from dotenv import load_dotenv
load_dotenv()


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--terms', nargs='+', default=[], help="search terms (e.g. \"carbon nanotube\") separated by spaces")
parser.add_argument('-l', '--limit', type=float, default=1e10, help="Search limit (e.g. 1e5) will be converted to int")


db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

args = parser.parse_args()
terms = args.terms
search_limit = int(args.limit)

if not os.path.exists('data'): os.mkdir('data')
fp_search_idxs = 'data/indexed_searches.json'
if os.path.exists(fp_search_idxs):
    with open(fp_search_idxs, 'r') as f:
        id_dict = json.load(f)
else:
    id_dict = {}

regexes = ['%{}%'.format(r) for r in terms]
for regex in regexes:

    print('Searching for regex: ' + regex)
    ids = nu.fileio.gen_ids_searchterm(
        con, 
        regex, 
        idx_name='id', 
        search_fields=['paperAbstract', 'title'], 
        search_limit=search_limit, 
        output_limit=1e10
    )
    # all_ids.append(ids)

    id_dict[regex] = ids

with open(fp_search_idxs, 'w') as f:
    json.dump(id_dict, f, indent=2)
