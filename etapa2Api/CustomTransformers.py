from sklearn.base import BaseEstimator, TransformerMixin
import contractions
import nltk

from nltk  import word_tokenize
import inflect
from nltk.corpus import stopwords
import unicodedata
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Punkt permite separar un texto en frases.
nltk.download('punkt_tab')

# Descarga todas las palabras vacias, es decir, aquellas que no aportan nada al significado del texto
nltk.download('stopwords')

# Descarga de paquete WordNetLemmatizer, este es usado para encontrar el lema de cada palabra
nltk.download('wordnet')

class ExpandContractions(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['expanded_text'] = X['Textos_espanol'].apply(contractions.fix)
        return X

class Tokenizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['tokenized_words'] = X['expanded_text'].apply(word_tokenize)
        return X

class NormalizeText(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.inflect_engine = inflect.engine()
        self.stop_words_spanish = set(stopwords.words('spanish'))
        self.stop_words_english = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['words_normalize'] = X['tokenized_words'].apply(self.normalize)
        return X

    def remove_non_ascii(self, words):
        new_words = []
        for word in words:
            new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
            new_words.append(new_word)
        return new_words

    def to_lowercase(self, words):
        new_words = []
        for word in words:
            new_words.append(word.lower())
        return new_words

    def remove_punctuation(self, words):
        new_words = []
        for word in words:
            new_word = re.sub(r'[^\w\s]', '', word)
            if new_word != '':
                new_words.append(new_word)
        return new_words

    def replace_numbers(self, words):
        new_words = []
        for word in words:
            if word.isdigit():
                new_word = self.inflect_engine.number_to_words(word)
                new_words.append(new_word)
            else:
                new_words.append(word)
        return new_words

    def remove_stopwords(self, words):
        new_words = []
        for word in words:
            if word not in self.stop_words_spanish and word not in self.stop_words_english:
                new_words.append(word)
        return new_words

    def normalize(self, words):
        words = self.remove_non_ascii(words)
        words = self.to_lowercase(words)
        words = self.remove_punctuation(words)
        words = self.replace_numbers(words)
        words = self.remove_stopwords(words)
        return words

class StemAndLemmatize(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['s&l'] = X['words_normalize'].apply(self.stem_and_lemmatize)
        return X

    def stem_words(self, words):
        stems = [self.stemmer.stem(word) for word in words]
        return stems

    def lemmatize_verbs(self, words):
        lemmas = [self.lemmatizer.lemmatize(word, pos='v') for word in words]
        return lemmas

    def stem_and_lemmatize(self, words):
        stems = self.stem_words(words)
        lemmas = self.lemmatize_verbs(words)
        return stems, lemmas

class CombineOriginalStemLemma(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['text_clean'] = X.apply(lambda row: ' '.join(self.combine_original_stem_lemma(row['words_normalize'], row['s&l'][0], row['s&l'][1])), axis=1)
        return X

    def combine_original_stem_lemma(self, original, stems, lemmas):
        combined = []
        for orig, stem, lemma in zip(original, stems, lemmas):
            if lemma.endswith('ar') or lemma.endswith('er') or lemma.endswith('ir'):
                combined.append(lemma)
            elif len(stem) < len(orig) / 2:
                combined.append(stem)
            else:
                combined.append(orig)
        return combined

class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit(self, X, y=None):
        try:
            self.vectorizer.fit(X['text_clean'])
            
        except:
            X=X.copy()
            X['text_clean'] = X['tokenized_words'].apply(lambda x: ' '.join(x))
            self.vectorizer.fit(X['text_clean'])

        return self

    def transform(self, X):
        try:
            X_tfidf = self.vectorizer.transform(X['text_clean'])
        except:
            X=X.copy()
            X['text_clean'] = X['tokenized_words'].apply(lambda x: ' '.join(x))
            X_tfidf = self.vectorizer.transform(X['text_clean'])

        return X_tfidf
