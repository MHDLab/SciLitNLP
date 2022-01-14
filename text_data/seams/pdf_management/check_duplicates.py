"""
Found that there seems to be some filenames that are duplicated in SEAM11 which
leads to each filename being offset 1 file from it's actual filepath in between
the duplicates. Want to point out duplicate filenames to catch all of those. 

Looks like it's only in seam 11. There are many duplicate titles throughout the seams. 
"""

#%%
import pandas as pd

import shutil
import os


seams_pdf_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs Archive\PDF Cleanup 2'
df_meta = pd.read_csv(os.path.join(seams_pdf_folder, 'SEAMS_metadata.csv'), index_col=['ID'])
# df_meta = df_meta.dropna(subset=['Filepath'])

df_meta['Title'] = df_meta['Title'].apply(str.strip)

df_meta = df_meta.where(df_meta['Title'] != 'Introduction').dropna(subset=['Title'])
df_meta = df_meta.where(df_meta['Title'] != 'Foreword').dropna(subset=['Title'])
df_meta = df_meta.where(df_meta['Title'] != 'Table Of Contents').dropna(subset=['Title'])
df_meta = df_meta.where(df_meta['Title'] != 'Cover Foreword And Introduction').dropna(subset=['Title'])

df_meta
#%%

# df_meta['Title']

df_meta.where(df_meta['Title'].duplicated(False)).dropna(subset=['Title'])


# for ID, row in df_meta.iterrows():
#     relative_fp = row['Filepath']
#     if relative_fp == relative_fp:
#         input_fp = os.path.join(seams_pdf_folder, relative_fp)

#         if not os.path.exists(input_fp):
#             print(input_fp)


#%%

df_meta.loc[372]['Title'] == df_meta.loc[381]['Title']

#%%

df_meta