"""
SEAMs text exploratory data analysis (EDA)
"""

#%%
import os 
import sqlite3
import matplotlib.pyplot as plt
import nlp_utils as nu

#%%
db_path = os.path.join(os.getenv('DB_FOLDER'), 'seams.db')
con = sqlite3.connect(db_path)
df = nu.fileio.load_df_SEAMs(con)

df = df.dropna(subset=['OCR_text'])
df.info()
# %%
df['year'].hist()
plt.ylabel('Counts')
plt.xlabel('Year')

#%%
df['processed_text'].str.len().hist(bins=50)
plt.xlabel('Number of characters')
plt.ylabel('Count')
# %%
df['processed_text'].str.split(' ').str.len().hist(bins=50)
plt.xlabel('Number of words')
plt.ylabel('Count')
# %%
