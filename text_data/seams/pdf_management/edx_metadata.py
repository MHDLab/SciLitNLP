#%%

import pandas as pd

import shutil
import os
import json
import requests

#%%

# # Get all metadata
# for i in range(3,35):
#     r = requests.get(r'https://edx.netl.doe.gov/dataset/package_metadata/seam-' + str(i), allow_redirects=True)
#     t = r.text

#     with open('edx_metadata/seam-' + str(i) + '.json', 'w') as f:
#         f.write(t)




#%%
import sqlite3
import nlp_utils as nu
db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
con = sqlite3.connect(db_path)
df_tm = nu.fileio.load_df_SEAMs(con).dropna(subset=['OCR_text'])

page_urls = pd.Series(index=df_tm.index, name='pdf_url')

json_folder = r'edx_metadata'

fps = [os.path.join(json_folder, fn) for fn in os.listdir(json_folder)]

for fp in fps:
    with open(fp, 'rb') as file:
        jobj = json.load(file)


    resources = jobj['resources']



    for doc in resources:
        doc_id = int(doc['name'].split('_')[0][2:])

        # print(doc['name'])
        # print(df_meta.loc[doc_id]['Title'])

        # url = doc['url']

        page_url = r'https://edx.netl.doe.gov/dataset/'
        page_url += doc['package_id']
        page_url += r'/resource/'
        page_url += doc['id']

        page_urls[doc_id] = page_url

page_urls.to_csv('edx_urls.csv')


# %%


