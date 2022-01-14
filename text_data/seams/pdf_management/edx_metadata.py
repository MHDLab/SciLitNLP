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

metadata_path = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs\SEAMs_metadata.csv'
df_meta = pd.read_csv(metadata_path, index_col=0)

page_urls = pd.Series(index=df_meta.index, name='pdf_url')

json_folder = r'C:\Users\aspit\Git\MHDLab-Projects\NLP_MHD\pdf_management\edx_metadata'

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

# %%


df_meta = pd.concat([df_meta, page_urls], axis=1)

#%%

df_meta.to_csv(metadata_path)
# %%
