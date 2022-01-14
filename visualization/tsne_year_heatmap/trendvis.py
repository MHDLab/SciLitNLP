#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sqlite3


from nlp_utils import fileio, sklearn_utils

db_folder = r'C:\Users\aspit\National Energy Technology Laboratory\MHD Lab - Documents\Publications\SEAMs'
df = fileio.load_df_SEAMs(db_folder).dropna(subset=['processed_text'])
# df = df.sample(300, random_state=42)
docs = df['processed_text']
texts = docs.apply(str.split)

#%%

import nlp_utils as nu
from nlp_utils.gensim_utils import basic_gensim_lda
from nlp_utils import gensim_utils
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

fixed_bigrams = None

pipeline = Pipeline([
('text_norm', nu.text_process.TextNormalizer()),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    # ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

texts_bigram = pipeline.fit_transform(texts)
tsne_x, tsne_y = sklearn_utils.calc_tsne(texts_bigram)
#%%

df['tsne_x'] = tsne_x
df['tsne_y'] = tsne_y

#%%



fig, ax = plt.subplots()


seams = list(set(df['SEAM']))


def get_tsne_axes(df, seam):
    df_sel = df.loc[df['SEAM'] == seam]
    return df_sel['tsne_x'],df_sel['tsne_y']

x, y = get_tsne_axes(df, 15)
scat = ax.scatter(x,y)
ax.set_title(str(15))

ax.set_xlim(-51,51)
ax.set_ylim(-51,51)

fig.tight_layout()
#%%

import matplotlib.animation as animation
plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\aspit\ffmpeg\bin\ffmpeg.exe'

from IPython.display import HTML

def plot(i):
    seam = seams[i]
    x, y = get_tsne_axes(df, seam)
    scat.set_offsets(np.c_[x,y])
    scat.axes.set_title('SEAM: ' + str(seam))


ani = animation.FuncAnimation(fig, plot, frames=len(seams))


HTML(ani.to_jshtml())

# with open("scatter_anim.html", "w") as f:
#     print(ani.to_html5_video(), file=f)





#%%
#https://python-graph-gallery.com/85-density-plot-with-matplotlib/

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde


# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins=150
k = kde.gaussian_kde([df['tsne_x'],df['tsne_y']])
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
xi, yi = np.mgrid[-51:51:nbins*1j, -51:51:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi/zi.max()
 
# Make the plot
qm = plt.pcolormesh(xi, yi, zi.reshape(xi.shape))
plt.show()


#%%

fig, ax = plt.subplots()


ax.set_xlim(-51,51)
ax.set_ylim(-51,51)
x, y = get_tsne_axes(df, 15)

# Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
nbins=50
k = kde.gaussian_kde([x,y])
# xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
xi, yi = np.mgrid[-51:51:nbins*1j, -51:51:nbins*1j]
zi = k(np.vstack([xi.flatten(), yi.flatten()]))
zi = zi/zi.max()
 
# Make the plot
qm = plt.pcolormesh(xi, yi, zi.reshape(xi.shape), axes=ax)
# plt.show()
# plt.hold(False)
#%%

def plot(i):
    seam = seams[i]
    x, y = get_tsne_axes(df, seam)

    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    nbins=50
    k = kde.gaussian_kde([x,y])
    xi, yi = np.mgrid[-51:51:nbins*1j, -51:51:nbins*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    zi = zi/zi.max()

    qm.set_array(zi.reshape(xi.shape)[:-1,:-1].ravel())

    ax.set_title('SEAM: ' + str(seam))


ani = animation.FuncAnimation(fig, plot, frames=len(seams))

HTML(ani.to_jshtml())

# with open("pcolor_anim.html", "w") as f:
#     print(ani.to_html5_video(), file=f)





# %%
