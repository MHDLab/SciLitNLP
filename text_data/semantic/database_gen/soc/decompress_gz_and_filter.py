#%%

from xopen import xopen
import os
import json
import pandas as pd
import gzip
import io
from tqdm import tqdm

import sqlite3

#Output sqlite database path 
# output_folder = r'/media/lee/Shared Storage/full'
output_folder = 'output'

if not os.path.exists('ouput_folder'): os.mkdir('ouput_folder')

# if os.path.exists('output/soc.db'): 
#     print("Removing existing database...")
#     os.remove('output/soc.db')

con = sqlite3.connect(os.path.join(output_folder, 'soc.db'))

#Temporary folder withfor downloaded .gz files
gz_folder = 'temp_gz'

#The semantic scholar dataset contains topic classificaitons from Microsoft Academic. The datset can be downselected to relevant topics.
# See the relative number of each paper here https://github.com/allenai/s2orc
# mag_keep = [
#     # 'Biology',
#     'Chemistry',
#     'Computer Science',
#     'Engineering',
#     'Physics',
#     'Materials Science',
#     'Mathematics',
#     'Economics',
#     'Geology',
#     'Environmental Science',
# ]


parse_files = [f for f in os.listdir(gz_folder) if '.gz' in f]

for metadata_file in parse_files:
    print('processing file ' + str(metadata_file))

    with gzip.open(os.path.join(gz_folder, metadata_file), 'rb') as gz:
        f = io.BufferedReader(gz)

        ds = []
        for line in tqdm(f.readlines()):
            metadata_dict = json.loads(line)
            # paper_id = metadata_dict['paper_id']

            if len(metadata_dict['inCitations']) == 0:
                continue
            if len(metadata_dict['outCitations']) == 0:
                continue

            if metadata_dict['paperAbstract'] == None:
                continue
            if len(metadata_dict['paperAbstract']) == 0:
                continue

            # mag_field_of_study = metadata_dict['fieldsOfStudy']

            # if mag_field_of_study:
            #     any_match = any(key in mag_field_of_study for key in mag_keep)

            #     if any_match:
            #         ds.append(metadata_dict)


            ds.append(metadata_dict)


        df = pd.DataFrame(ds)
        df = df.where(df['paperAbstract'].isnull() == False).dropna(how='all') #Think this is already covered above now.

        #When redownloading test soc dataset, got data with missing years that messes up conversion to int. 
        df['year'] = df['year'].astype(float)
        
        na_years = df['year'].isna().sum()
        if na_years > 0:
            print("Found {} records with missing years, dropping".format(na_years))
            df = df.dropna(subset=['year'])
        
        df['year'] = df['year'].astype(int)

        df_out = df.set_index('id')

        df_out = df_out.applymap(str)

        df_out = df_out[[
            'title',
            'paperAbstract',
            'inCitations',
            'outCitations',
            'year',
            's2Url',
            'doi',
            'fieldsOfStudy',
            'magId'
        ]]

        df_out.to_sql('raw_text', con, if_exists='append')

con.close()


# %%
