
import nlp_utils as nu
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from corextopic import corextopic as ct
from dotenv import load_dotenv
load_dotenv()


def corex_pipeline(docs, stopwords, corex_anchors, fixed_bigrams, n_hidden):
    pipeline = Pipeline([
    ('text_norm', nu.text_process.TextNormalizer(post_stopwords=stopwords)),
    ('bigram', nu.gensim_utils.Gensim_Bigram_Transformer(bigram_kwargs={'threshold':20, 'min_count':10}, fixed_bigrams=fixed_bigrams)),
    ('vectorizer', CountVectorizer(max_features=None, min_df=0.001, max_df = 0.5, tokenizer= lambda x: x, preprocessor=lambda x:x, input='content')), #https://stackoverflow.com/questions/35867484/pass-tokens-to-countvectorizer
])

    texts = docs.apply(str.split)
    X = pipeline.fit_transform(texts)
    feature_names = pipeline['vectorizer'].get_feature_names()

    topic_model = ct.Corex(n_hidden=n_hidden, seed=42)  # Define the number of latent (hidden) topics to use.
    topic_model.fit(X, words=feature_names, docs=docs.index, anchors=corex_anchors, anchor_strength=5)
    return topic_model
