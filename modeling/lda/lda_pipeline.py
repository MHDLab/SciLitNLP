import nlp_utils as nu
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv
load_dotenv()

from nlp_utils.gensim_utils import basic_gensim_lda

def lda_bigram_pipeline(texts, stopwords, fixed_bigrams, bigram_kwargs, lda_kwargs):
    pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=stopwords)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs, fixed_bigrams=fixed_bigrams)),
])

    texts_bigram = pipeline.fit_transform(texts)


    lda_model = basic_gensim_lda(texts_bigram, lda_kwargs)
    return lda_model


