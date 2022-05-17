import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import glob
from nltk.corpus import words
from nltk.corpus import wordnet
from sklearn.utils.extmath import randomized_svd as svd
from sentiment_analysis import read_file

# nltk.download("words")

stop_words = stopwords.words("english")
stop_words.extend(set(string.punctuation))
words = set(words.words() + list(wordnet.words()))
# stemmer = nltk.stem.PorterStemmer()
stemmer = nltk.stem.SnowballStemmer("english")
meta_df = pd.read_csv("meta_data.csv")
ticker_list = set(meta_df.Ticker.to_list())
stop_words.extend(ticker_list)
stop_words.extend([x.lower() for x in ticker_list])


class myTokenizer:
    ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`"]

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        tokens = [
            self.wnl.lemmatize(t)
            for t in word_tokenize(doc)
            if t not in self.ignore_tokens
        ]
        return tokens


# Lemmatize the stop words
tokenizer = myTokenizer()
token_stop = tokenizer(" ".join(stop_words))


def get_corpus(path_list):
    corpus = []
    for path in path_list:
        text = read_file(path)
        text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub("\d+", "", text)
        text = " ".join(w for w in nltk.word_tokenize(text) if w.lower() in words)
        corpus.append(text)
    return corpus


def get_mat(corpus, bigram=True):
    if bigram:
        vect = TfidfVectorizer(
            lowercase=True,
            stop_words=token_stop,
            tokenizer=tokenizer,
            ngram_range=(1, 2),
        )
        vects = vect.fit_transform(corpus)
        td = pd.DataFrame(vects.todense())
        td.columns = vect.get_feature_names_out()
        term_document_matrix = td.T
    else:
        vect = TfidfVectorizer(
            lowercase=True,
            stop_words=token_stop,
            tokenizer=tokenizer,
            ngram_range=(1, 1),
        )
        vects = vect.fit_transform(corpus)
        td = pd.DataFrame(vects.todense())
        td.columns = vect.get_feature_names_out()
        term_document_matrix = td.T
    return term_document_matrix


def svd_tfidf_matrix(matrix, n):
    mat = matrix.to_numpy()
    V_T = svd(mat, n_components=n, n_iter=5, random_state=42)[2]
    return V_T


if __name__ == "__main__":
    root_path = Path.cwd()
    earnings_call_path = glob.glob(f"{root_path}/*/*/*/earnings_call.txt")
    corpus = get_corpus(earnings_call_path)
    term_doc_matrix = get_mat(corpus=corpus, bigram=False)
    svd_mat = svd_tfidf_matrix(term_doc_matrix, 4)
    df = pd.DataFrame(
        {
            "Path": earnings_call_path,
            "svd_1": svd_mat[0],
            "svd_2": svd_mat[1],
            "svd_3": svd_mat[2],
            "svd_4": svd_mat[3],
        }
    )
    df.Path = df.Path.apply(lambda x: x.strip("earnings_call.txt"))
    df.to_csv(f"{root_path}/svd_df.csv")
    print(f"output csv to {root_path}/!")
