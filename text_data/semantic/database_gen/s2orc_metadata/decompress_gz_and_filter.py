#%%

from xopen import xopen

import os
import json

import pandas as pd




#%%

import gzip
import io
from tqdm import tqdm

import sqlite3

con = sqlite3.connect(r'C:\Users\aspit\Git\MLEF-Energy-Storage\s2orc\data\full\db_s2orc.db')
METADATA_DIR = r'C:\Users\aspit\Git\MLEF-Energy-Storage\s2orc\data\full\metadata'

mag_keep = [
    'Biology',
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

            if metadata_dict['has_inbound_citations'] == 0.0:
                continue

            if metadata_dict['abstract'] == None:
                continue

            mag_field_of_study = metadata_dict['mag_field_of_study']

            if mag_field_of_study:
                any_match = any(key in mag_field_of_study for key in mag_keep)

                if any_match:
                    ds.append(metadata_dict)

        df = pd.DataFrame(ds)
        df = df.where(df['abstract'].isnull() == False).dropna(how='all') #Think this is already covered above now.

        df_out = df.set_index('paper_id')

        df_out = df_out.applymap(str)

        df_out.to_sql('raw_text', con, if_exists='append')

con.close()

