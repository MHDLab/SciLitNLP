
"""
This script tests the parameters for the word2vec model
https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html
https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
"""

#%%
"""import necessary libraries"""
from gensim.models import Word2Vec

import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from gensim.models.phrases import Phrases, Phraser
from nlp_utils import text_analysis
from sklearn.decomposition import PCA
#%%
"""load the data"""
from nlp_utils import gensim_utils, sklearn_utils, fileio
data_folder = r'C:\Users\byahn\Code\MLEF\ES_TextData\data'
df = fileio.load_df(os.path.join(data_folder, 'SOC_ES.db'))
# df = df.sample(500)

#%%
"""get texts"""
texts = df['processed_text'].values
texts = [t.split() for t in texts]


#%%
"""Generate Bigrams"""
print("Generating Bigrams")

phrases= Phrases(texts, min_count = 10, threshold = 100)

bigram = Phraser(phrases)

texts = bigram[texts]

#%%
"""Get top words list"""

tw = text_analysis.top_words(texts, num_words=50)

print([w[0] for w in tw])

# list(sentences)
# %%
"""scan model parameters"""
f = open("parameter_testing.txt", "w+")
s = 0
h = 0
for w in range(5,21,5):
	for n in range(5,21,5):
		mod = Word2Vec(sg=0, seed=42, min_count = 100, window= w, vector_size=100, negative = n, hs=0)
		mod.build_vocab(texts)
		print("Training Model")
		mod.train(texts, total_examples=mod.corpus_count, epochs=30)

		top_words = ['energi', 'storag', 'system', 'power', 'use', 'batteri', 'high', 'control', 'materi', 'electr', 'perform', 'base', 'gener', 'applic', 'heat', 'thermal', 'model', 'c', 'result', 'oper', 'devic', 'studi', 'develop', '...', 'method', 'propos', 'design', 'effect', 'electrod', 'paper', 'voltag', 'effici', 'charg', 'optim', 'grid', 'capac', 'load', 'current', 'structur', 'present', 'cost', 'temperatur', 'solar', 'cycl', 'li', 'increas', 'ion', 'densiti', 'provid', 'also']
		embedded_words = mod.wv[top_words]
		pca= PCA(n_components=2)
		result = pca.fit_transform(embedded_words)
		nearest_vocab = []
		for word in top_words:
			s=''
			nearest_vocab.append(s.join((t[0] + ', ') for t in mod.wv.most_similar(word)))

		pc1 = result[:,0].tolist()
		pc2 = result[:,1].tolist()

		fig, ax = plt.subplots()
		ax.scatter(pc1, pc2)
		i=0

		for txt in top_words:
			ax.annotate(text=txt, xy=(pc1[i], pc2[i]))
			i+=1
		title_string = 's0' + '_h' + str(h) + '_w' + str(w) + '_n' + str(n)
		ax.set_title(title_string)
		plt.savefig(fname=title_string)

		f.write(title_string)
		f.write("colio2 - cathod + anod = " + str(mod.wv.most_similar(positive=['colio2', 'anod'], negative=['cathod'])[0])+"\n")
		f.write("batteri + h - li = " + str(mod.wv.most_similar(positive=['batteri', 'h'], negative=['li'])[0])+'\n')
		f.write("molten_salt + chemic - thermal = " + str(mod.wv.most_similar(positive=['molten_salt', 'chemic'], negative=['thermal'])[0]) + "\n")
		f.write("cost + cycl - effici = " + str(mod.wv.most_similar(positive=['cost', 'cycl'], negative=['effici'])[0])+"\n")
		f.write("overpotenti + g-1 - ma_cm2 = " + str(mod.wv.most_similar(positive=['overpotenti', 'g-1'], negative=['ma_cm2'])[0]) + "\n")

		f.write('similar words to electrod' + str(mod.wv.most_similar('electrod')) +"\n\n")
f.close()




# %%
