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
parser.add_argument('-r', '--regex', type=str, help="search python regex that the search fields contain eg. \"carbon nanotube\" or \"oxygen (evolution|reduction)\", Note that \\b (word boundaries) are added to the prefix and and suffix of the regex")
parser.add_argument('-sl', '--search-limit', type=float, default=0, help="Search limit (e.g. 1e5) will be converted to int. 0 for no limit.")
parser.add_argument('-ol', '--output-limit', type=float, default=0, help="Output limit (e.g. 1e5) will be converted to int. 0 for no limit.")
parser.add_argument('-p', '--purge', action='store_true', help="delete all existing searches")


db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

args = parser.parse_args()
regex = args.regex
search_limit = int(args.search_limit)
output_limit = int(args.output_limit)


if not os.path.exists('data'): os.mkdir('data')
fp_search_idxs = 'data/indexed_searches.json'
if os.path.exists(fp_search_idxs) and not args.purge:
    with open(fp_search_idxs, 'r') as f:
        id_dict = json.load(f)
else:
    id_dict = {}


print('Searching for regex: ' + regex)
ids = nu.fileio.gen_ids_regex(
    con, 
    regex, 
    idx_name='id', 
    search_fields=['paperAbstract', 'title'], 
    search_limit=search_limit, 
    output_limit=output_limit
)
# all_ids.append(ids)



#Downselect based on paper being english
import langdetect
print("dropping languages other than english")

df = nu.fileio.load_df_semantic(con, ids)
docs = df['title'] + ' ' + df['paperAbstract']

langs = docs.apply(langdetect.detect)
docs = docs.where(langs == 'en').dropna()

print(langs.value_counts())
print(" df size after: {} ".format(len(docs)))

ids = docs.index.tolist()

id_dict[regex] = ids

with open(fp_search_idxs, 'w') as f:
    json.dump(id_dict, f, indent=2)
