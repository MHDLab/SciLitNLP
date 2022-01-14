#%%
import pandas as pd
import sqlite3
import os
import ast


import sys
sys.path.append('nlp_utils')
from nlp_utils.textprocess import tokenizer, flag

import git
repopath = git.Repo('.', search_parent_directories=True).working_tree_dir
data_folder = os.path.join(repopath, 'data')



con = sqlite3.connect(os.path.join(data_folder, 'seamsnlp_final.db'))
cursor = con.cursor()

df_text = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID').dropna(subset=['OCR_text'])
#%%

#TODO: move to a file dictionary? 
with open(os.path.join(data_folder,"autocorrect.txt"), encoding='utf-8') as f:
    custom_autocorrect = f.read().splitlines()

with open(os.path.join(data_folder,"english_stopwords.txt"), encoding='utf-8') as f:
    custom_stop_words = f.read().splitlines()

with open(os.path.join(data_folder,"skip_words.txt"), encoding='utf-8') as f:
    skip_words = f.read().splitlines()

with open(os.path.join(data_folder,"dictionary.txt"), "r") as f:
    contents = f.read()
    lookup_dict = ast.literal_eval(contents)

#%%
#TODO: write method to add column with specified data type

cursor.execute('select * from texts')

names = list(map(lambda x: x[0], cursor.description))

if 'processed_text' not in names:
    print("creating processed_text column")
    cursor.execute("ALTER TABLE texts ADD COLUMN processed_text TEXT")

if 'misspelled' not in names:
    print("creating misspelled column")
    cursor.execute("ALTER TABLE texts ADD COLUMN misspelled TEXT")



#%%
ids = df_text.dropna(subset=['OCR_text']).index

# ids = [id for id in ids if id in range(1828,1829)]

for id in ids:
    text = df_text['OCR_text'][id]
    print('processing file: ' + str(id))

    mytokens1 = tokenizer(text, lookup_dict, custom_stop_words, skip_words, custom_autocorrect)
    processed_text, misspelled = flag(mytokens1, custom_stop_words, skip_words)

    query = """UPDATE texts SET processed_text= (?) WHERE ID = (?)"""
    cursor.execute(query, (processed_text, id))

    query = """UPDATE texts SET misspelled= (?) WHERE ID = (?)"""
    cursor.execute(query, (misspelled, id))

    con.commit()
# %%
