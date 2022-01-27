"""
Exploratory data analysis of the semantic scholar dataset. 
"""

#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3
import nlp_utils as nu
from dotenv import load_dotenv
load_dotenv()

db_path = os.path.join(os.getenv('DB_FOLDER'), 'soc.db')
con = sqlite3.connect(db_path)

#Get the ideas for the first num_papers and return df
num_papers = 1e4
cursor = con.cursor()
cursor.execute("SELECT id FROM raw_text LIMIT {}".format(num_papers))
results = cursor.fetchall()
ids = [t[0] for t in results]

df = nu.fileio.load_df_semantic(con, ids)


fig_path = r'output'
if not os.path.exists(fig_path): os.mkdir(fig_path)

# %%
plt.figure(1)
fos = df['fieldsOfStudy'].str.strip('][').str.split(', ').apply(lambda x: x[0])
barlist = fos.value_counts(ascending=False).plot(kind='bar', figsize=(15,5), title="Distribution of Abstract Field of Study", ylabel="Number of Abstracts (54,125 total)", color=['#6DA367']*len(fos.value_counts()), fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(fig_path,'FOS.png'), bbox_inches="tight")

#fos.hist(figsize=(15,5))
#plt.xticks(rotation=45)
# %%

plt.figure(2)
df['year'].dropna().astype(int).value_counts().sort_index(ascending=False).plot(marker='o', figsize=(15,5), title="Years of Abstract Publications", ylabel="Number of Abstracts (54,125 total)", color=['#6DA367']*len(df['year'].dropna().astype(int).value_counts()), fontsize=15)
plt.savefig(os.path.join(fig_path,'Years.png'), bbox_inches="tight")

# %%

plt.figure(3)
df['paperAbstract'].dropna().apply(len).hist(bins = 100, figsize=(5,5), color="#6DA367")
plt.title('Distribution of Abstract Length')
plt.ylabel('Number of Abstracts')
plt.xlabel('Length of Abstract (characters)')
plt.xlim(0,8000)
plt.xticks(np.arange(0,8000, step=1000))
plt.savefig(os.path.join(fig_path,'Length.png'), bbox_inches="tight")

#%%
plt.figure(4)
bins = np.logspace(np.log10(0.9),np.log10(20000), 100)
df['inCitations'].apply(len).value_counts().hist(bins=bins, figsize=(5,5), color="#6DA367")

plt.title('Distribution of Paper Citations')
plt.ylabel('Number of Papers')
plt.xlabel('Number of citations in literature')
plt.yscale('log')
plt.xscale('log')
plt.savefig(os.path.join(fig_path,'InCite.png'), bbox_inches="tight")

# %%
plt.figure(5)
bins = np.logspace(np.log10(0.9),np.log10(20000), 100)
df['outCitations'].apply(len).value_counts().hist(bins=bins,figsize=(5,5), color="#6DA367")

plt.title("Distribution of References")
plt.ylabel('Number of Papers')
plt.xlabel('Number of Citations in each Paper')
plt.yscale('log')
plt.xscale('log')
plt.savefig(os.path.join(fig_path,'OutCite.png'), bbox_inches="tight")
# %%
