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
parser.add_argument('-l', '--limit', type=float, default=1e10, help="Search limit (e.g. 1e5) will be converted to int")
parser.add_argument('-p', '--purge', action='store_true', help="Search limit (e.g. 1e5) will be converted to int")


db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

args = parser.parse_args()
regex = args.regex
search_limit = int(args.limit)

regex = "\\b" + regex + "\\b" #Assume that we want word boundaries (look for the regex anywhere, I think)

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
    output_limit=1e10
)
# all_ids.append(ids)

#Remove the word boundaries for the json dictionary. #TODO: some package (I think tmtoolkit) was causing issues being able to use pyhton 3.9, which would simplify this
remove_str = "\\b"
# regex = regex[len(remove_str):]
# regex = regex[:len(remove_str)]
regex = regex.strip(remove_str)

id_dict[regex] = ids

with open(fp_search_idxs, 'w') as f:
    json.dump(id_dict, f, indent=2)
