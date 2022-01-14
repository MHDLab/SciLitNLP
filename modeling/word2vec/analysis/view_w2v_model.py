# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# This notebook looks at the Word2Vec model generatied in word_embed_genmodel.py

# %%
import os
from gensim.models import Word2Vec

w2v_models_folder = r'C:\Users\aspit\Git\MLEF-Energy-Storage\ES_W2V\w2v_models'

mod = Word2Vec.load(os.path.join(w2v_models_folder, 'word2vec_semantic.model'))

# %% [markdown]
# ## Word similarity
# 
# In a word2vec model each word is represented as a vector. I understand the axes/components of this vector space as abstract 'meanings'
# 
# we check most similar words to a given word by finding words that are the closest in the vector space. 

# %%

mod.wv.most_similar('battery')

# %% [markdown]
# The basic application with word vectors is to be able to quantify the relationships between words. The most often used example is finding the word that has the same relationship to 'man' as 'queen' has to 'woman' (king). 
# 
# https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/
# 
# We might be able to use this to find relatonships between different technologies or physical concepts. I attempt this below without much sucess to try and find the material that is used in Li-ion battery anodes from the material that is used in cathodes. This is inspired by the Tshitoyan 2019 paper. 
# 
# However, here it seems that the words are being dominated by one word. I.e. the negative affect of cathode isn't doing much and the results are just dominated by 'licoo2'. 

# %%
mod.wv.most_similar(
    positive = ['licoo2', 'anode'],
    negative = ['cathode']
)[0:5]


# %%
mod.wv.most_similar(
    positive = ['graphite', 'cathode'],

    negative = ['anode']
)[0:5]


# %%
mod.wv.most_similar(
    positive = ['solar', 'fossil'],
    negative = ['battery']
)

# %% [markdown]
# ## PCA word visualization
# 
# We can visualize a set of words together by projecting their vectors into a 2D plane using Principal Components Analysis. PCA is commonly used to visualize higher dimensional datasets. 

# %%
#We can check if a word is in the vocabulary this way
'system' in mod.wv


# %%
#Generate a vector representation (X) of a collection of words

top_words = ['system', 'power', 'battery', 'high', 'performance', 'application', 'material', 'device', 'cost', 'control', 'capacity', 'grid', 'model', 'ion', 'density', 'electrode', 'renewable', 'technology', 'design', 'electric', 'generation', 'cycle', 'time', 'low', 'supercapacitor', 'load', 'source', 'show', 'carbon', 'hybrid', 'operation', 'current', 'strategy', 'simulation', 'different', 'voltage', 'efficiency', 'cell', 'lithium', 'large', 'electrochemical', 'demand', 'potential', 'vehicle', 'thermal', 'structure', 'rate', 
'solar', 'process', 'temperature']

X = mod.wv[top_words]


# %%
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
chemical = ['hydrogen','ammonia','methane','methanol', 'methane', 'fossil']

electrochemical = ['lithium', 'ion', 'lead_acid', 'redox_flow', 'vanadium']

mechanical = ['compressed_air', 'laes', 'gravitational', 'pumped_hydro', 'flywheel']

thermal = ['thermal', 'latent']

electrical = ['supercapacitor', 'smes']

all_words = [*chemical, *electrochemical, *mechanical, *thermal, *electrical]

X = mod.wv[all_words]

pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
plt.figure(figsize=(10,10))
plt.scatter(result[:, 0], result[:, 1])
# words = list(model.wv.vocab)
for i, word in enumerate(all_words):
	plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()


# %%



