import re

import pandas as pd
import spacy
from sklearn.base import TransformerMixin
from tqdm import tqdm

tqdm.pandas()
spacy_model = spacy.load('ru_core_news_sm')


class TextTransformer(TransformerMixin):
    def __init__(self):
        self.spacy_model = spacy_model
        self.ignore = ['PUNCT', 'SYM', 'NUM', 'PROPN']

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        spacy_text = X.progress_apply(spacy_model)
        X = spacy_text.progress_apply(
            lambda text: ' '.join([word.lemma_ for word in text if word.pos_ not in self.ignore])
        )
        X = pd.Series(X.progress_apply(lambda text: self.__clear_digits(text)))
        return X

    def __clear_digits(self, text):
        return re.sub('(\d+ )', '', text)
