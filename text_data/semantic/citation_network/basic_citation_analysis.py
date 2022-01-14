

#%%
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import os

DATASET_DIR = r'C:\Users\aspit\Git\NLP-Semantic\datasets'
db_path = os.path.join(DATASET_DIR, 'soc.db')

con = sqlite3.connect(db_path)
cursor = con.cursor()

#%%
# Energy Storage Lit analysis


all_dois = pd.read_csv('data/tech_doi.csv')
input_dois = all_dois['general'].dropna()

df = load_df_semantic(con, input_dois, cust_idx_name='doi')

for doi in input_dois:
    if doi not in df['doi'].values:
        print("Could not find doi: " + str(doi))

df

#%%

from nlp_utils.citation import get_citations, build_citation_community
df_com = build_citation_community(df, con, n_iter=2, frac_keep_factor=0.5, n_initial_trim=200)


#%%

df_com['outCitations'].apply(len).hist(bins=100)
plt.yscale('log')
plt.xscale('log')

#%%

df_com.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()

#%%

#how many papers have the term 'review'
review_papers = (df_com['title'] + df_com['paperAbstract']).str.contains('review')
review_papers.value_counts()

#%%

df_review = df_com.where(review_papers==True).dropna(how='all')
df_review['outCitations'].apply(len).hist(bins=1000, alpha=0.5)
df_not_review = df_com.where(review_papers==False).dropna(how='all')
df_not_review['outCitations'].apply(len).hist(bins=1000, alpha=0.5 )

plt.yscale('log')
plt.xscale('log')
#%%

df_not_review.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()

#%%

df_not_review = df_not_review.where(df_not_review['outCitations'].apply(len) < 100).dropna(how='all')

df_not_review.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()



#%%


# cursor.execute("SELECT id from raw_text ORDER BY RANDOM() LIMIT 100")
cursor.execute("SELECT id FROM raw_text LIMIT 10000")
results = cursor.fetchall()

ids = [t[0] for t in results]
#%%

from nlp_utils.fileio import load_df_semantic

df = load_df_semantic(con, ids)

#%%

df['outCitations'].apply(len).hist(bins=100)
plt.yscale('log')
plt.xscale('log')


#%%


df.groupby('year').apply(
    lambda x: x['outCitations'].apply(len).mean()
).plot()



#%%
df['incits_year'] = df['inCitations'].apply(len)/df['years_ago']

df['incits_year'].hist(bins=1000)
plt.yscale('log')
plt.xscale('log')
# df

#%%

# I think making a citation map skews the results to a higher average citations per year. 
# TODO: check this. 

df_keep = df.where(df['incits_year'] > 1).dropna(how='all')
df_keep

# %%

# Drop any articles from the citation map
# This ended up droping a tiny fraction 

### df_temp.index.to_series().to_csv('data/cit_map_ids.csv') # line used to output this file

# cit_map_ids = pd.read_csv(r'data/cit_map_ids.csv')['id']
# # idx_keep = [id for id in df.index if id not in cit_map_ids]
# # len(idx_k)

# df = df.drop(index=cit_map_ids, errors='ignore')

#%%


# %%
