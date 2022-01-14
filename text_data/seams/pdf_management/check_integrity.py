#%%

import pandas as pd

import shutil
import os


seams_pdf_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs\Final'
df_meta = pd.read_csv(os.path.join(seams_pdf_folder, 'SEAMS_metadata.csv'), index_col=['ID'])
# df_meta = df_meta.dropna(subset=['Filepath'])

#%%


for ID, row in df_meta.iterrows():
    relative_fp = row['Filepath']
    if relative_fp == relative_fp:
        input_fp = os.path.join(seams_pdf_folder, relative_fp)

        if not os.path.exists(input_fp):
            print(input_fp)



# %%
