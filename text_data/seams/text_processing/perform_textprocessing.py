#%%
import pandas as pd
import sqlite3
import os
import ast
import nlp_utils as nu
from dotenv import load_dotenv
load_dotenv()

from utils import tokenizer, flag


db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
con = sqlite3.connect(db_path)
cursor = con.cursor()
df_text = nu.fileio.load_df_SEAMs(con).dropna(subset=['OCR_text'])
#%%

data_folder = 'settings'
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

# ids = [id for id in ids if id in range(1825,1829)]

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
# # %%
