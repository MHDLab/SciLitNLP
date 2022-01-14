"""
This code examines the relationships between groups of words. I didn't really
get good results before trying to do word vector math with single words so
wanted to try doing math with collections of words, essentially trying to work
with the average vector of those words.

The key idea of the whitepaper is that storage technologies can be split up
based on storage medium and energy transformation. So it would be great to find
some relationship between collection of storage media, transformations, and
technologies.

I didn't get great results and want to explore further. My guess is that the
'meanings' of words are so caught up in subfield jargon in the abstract that the
physical relationship between overarchinng concepts will be obsured, but need to
explore further.
"""

#%%

from gensim.models import Word2Vec

mod = Word2Vec.load('w2v_models/word2vec.model')

#This is how you read directly from the databases to get texts (instead of the csv files generated for the other repos)
from nlp_utils import gensim_utils, sklearn_utils, fileio
data_folder = r'C:\Users\aspit\Git\MLEF-Energy-Storage\ES_TextData\data'
df = fileio.load_df(os.path.join(data_folder, 'nlp_justenergystorage_100.db'))
texts = df['processed_text'].values
texts = [t.split() for t in texts]

#%%
from nlp_utils import text_analysis
tw = text_analysis.top_words(texts, num_words=50)

top_words = [w[0] for w in tw]

print(top_words)


#%%

X = mod.wv[top_words]

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#Reduce dimensions of X with PCA
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# create a scatter plot of the projection
plt.figure(figsize=(10,10))
plt.scatter(result[:, 0], result[:, 1])

for i, word in enumerate(top_words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
# %%
mod.wv.most_similar(
    positive = ['solar', 'fossil'],
    negative = ['battery']
)

#%%

mod.wv.most_similar(['heat', 'thermal'],topn=30)




#%%


"""
Define collections of words, this was done with an iterative process looking at the resulting similar words
"""

thermal_storage_words = [
    'pcm',
    'pcms',
    'sensible',
    'latent',
    'latent_heat',
    'sensible',
    'sensible_latent',
    'rock_bed',
    'gravel',
    'packed_bed',
    'hot_water',
]

mod.wv.most_similar(thermal_storage_words,topn=30)

#%%

thermomechanical_words =[
    'compressor',
    'expander',
    'isentropic',
    'adiabatic',
    'diabatic',
    'isobaric',
    'isothermal',
    'compress',
    'compression',
    'compressor_expander',
    'compression_expansion',
    'turbomachinery'
]

mod.wv.most_similar(thermomechanical_words,topn=30)


#%%

thermal_tech_words = [
    'caes',
    'aa_caes',
    'huntorf',
    'uwcaes',
    'ptes',
    'compressed_air',
    'laes',
    'air_caes',
    'adiabatic_compressed'
]

mod.wv.most_similar(thermal_tech_words,topn=30)

#%%

mod.wv.most_similar(positive= [*thermal_storage_words, *thermomechanical_words],topn=30)
# %%
mod.wv.most_similar(
    positive= [*thermal_storage_words, *thermomechanical_words],
    negative= thermal_tech_words,
    topn=30)


#%%

chemical_storage_words = [
    'hydrogen',
    'ammonia',
    'h2',
    'methanol',
    'ch4',
    'methane',
    'lithium',
    'li',
    'v2o5',
    'vanadium'
]

mod.wv.most_similar(
    positive= chemical_storage_words,
    topn=30)


#%%

electrochemical_words = [
    'electrochemical',
    'electroreduction',
    'oxidation',
    'electrode',
    'cathode',
    'anode',
    'electrolysis',
    'catalyst',
    'reaction',
    'redox',
    'electrocatalytic',
    'electrocatalyst'
]

mod.wv.most_similar(
    positive= electrochemical_words,
    topn=30)

#%%

mod.wv.most_similar(
    positive= [*thermal_storage_words, *electrochemical_words],
    negative= thermomechanical_words,
    topn=30)

#%%

mod.wv.most_similar(
    positive= [*thermomechanical_words,  *chemical_storage_words],
    negative= [*thermal_storage_words],
    topn=30)




#%%

"""
First form the average vector before doing math.
"""

import numpy as np

X_electrochemical = np.mean(mod.wv[[*chemical_storage_words]], axis=0)


mod.wv.similar_by_vector(X_electrochemical)

#%%

X_thermal = np.mean(mod.wv[[*thermal_storage_words]], axis=0)

mod.wv.similar_by_vector(X_thermal)


#%%


X_thermal_tech = np.mean(mod.wv[thermal_tech_words], axis=0)

mod.wv.similar_by_vector(X_thermal_tech)


#%%



mod.wv.similar_by_vector(X_thermal_tech - X_thermal + X_electrochemical)
# %%
