import pandas as pd
import os
import sqlite3

metadata_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs\Final'
df = pd.read_csv(os.path.join(metadata_folder, 'SEAMS_metadata.csv'), index_col=0)


db_folder = r'C:\Users\aspit\Git\MHDLab-Projects\NLP_MHD\data'

con = sqlite3.connect(os.path.join(db_folder, 'seamsnlp_final.db'))
cursor = con.cursor()

df.to_sql('metadata', con=con)

empty_text_table = pd.DataFrame(index=df.index)

empty_text_table.to_sql('texts', con=con)