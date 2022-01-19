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
con = sqlite3.connect(r'C:\Users\aspit\Git\MLEF-Energy-Storage\semantic_opencorpus\data\soc.db')

#Temporary folder for downloaded .gz files
METADATA_DIR = r'C:\Users\aspit\Git\MLEF-Energy-Storage\semantic_opencorpus\data\destinationPath'

#The semantic scholar dataset contains topic classificaitons from Microsoft Academic. The datset can be downselected to relevant topics.
# See the relative number of each paper here https://github.com/allenai/s2orc
mag_keep = [
    # 'Biology',
    'Chemistry',
    'Computer Science',
    'Engineering',
    'Physics',
    'Materials Science',
    'Mathematics',
    'Economics',
    'Geology',
    'Environmental Science',
]


parse_files = [f for f in os.listdir(METADATA_DIR) if '.gz' in f]

for metadata_file in parse_files:
    print('processing file ' + str(metadata_file))

    with gzip.open(os.path.join(METADATA_DIR, metadata_file), 'rb') as gz:
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

            mag_field_of_study = metadata_dict['fieldsOfStudy']

            if mag_field_of_study:
                any_match = any(key in mag_field_of_study for key in mag_keep)

                if any_match:
                    ds.append(metadata_dict)

        df = pd.DataFrame(ds)
        df = df.where(df['paperAbstract'].isnull() == False).dropna(how='all') #Think this is already covered above now.

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
