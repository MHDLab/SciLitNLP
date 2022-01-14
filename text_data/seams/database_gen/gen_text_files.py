#%%

import pandas as pd
import os
import git
import sqlite3


#%%

repopath = git.Repo('.', search_parent_directories=True).working_tree_dir
data_folder = os.path.join(repopath, 'data')
out_folder = os.path.join(data_folder, 'OCR Text Files')

if os.path.exists(out_folder):
    for file in os.listdir(out_folder):
        os.remove(os.path.join(out_folder, file))
else:
    os.mkdir(out_folder)

#%%

con = sqlite3.connect(os.path.join(data_folder, 'seamsnlp_final.db'))
cursor = con.cursor()
df = pd.read_sql_query("SELECT * FROM texts", con, index_col='ID')
df_meta = pd.read_sql_query("SELECT * FROM metadata", con, index_col='ID')

df = pd.concat([df, df_meta], axis=1)

# %%



df = df.dropna(subset=['OCR_text'])

for id in df.index:
    text_out = df['OCR_text'][id]

    paper_title_str = df['Title'][id][0:30].replace(r'/','').replace('\"','')

    title = 'SEAM' + str(df['SEAM'][id]) + '_' + str(id) + '_' + paper_title_str +'.txt'

    out_path = os.path.join(out_folder, title)


    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(text_out)
    # break

#%%
text_out