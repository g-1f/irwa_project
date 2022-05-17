import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from utility import util_preprocess_text, strip_links, strip_all_entities
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pathlib import Path
import tqdm.notebook as tq
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import glob

stop_words = stopwords.words("english")

meta_df = pd.read_csv("meta_data.csv")
ticker_list = set(meta_df.Ticker.to_list())
url_list = meta_df.Earnings_url.to_list()
path_list = meta_df.Path.to_list()
path_d = dict(zip(meta_df.Earnings_url, meta_df.Path))

stop_words.extend(ticker_list)
stop_words.extend([x.lower() for x in ticker_list])


def read_file(path):
    with open(path) as f:
        contents = f.readlines()
    return "".join(contents)


def get_corpus(path_list):
    corpus = []
    for path in path_list:
        text = read_file(path)
        corpus.append(text)
    return corpus


def count_vect_filter(corpus):
    count_vectorizer = CountVectorizer(preprocessor=lambda x: re.sub(r"(\d)+", "", x))
    stop_words = count_vectorizer.fit_transform(corpus)
    count = np.sum(stop_words.toarray(), axis=0).tolist()
    names = count_vectorizer.get_feature_names_out()
    count_df = pd.DataFrame(count, index=names, columns=["count"])
    count_df = count_df.reset_index()
    count_df = count_df[count_df["count"] > 1]
    filter_list = count_df["index"].to_list()
    return filter_list


def preprocessing(path, filter_list):
    preprocessed_list = []
    earning_call_path = "/earnings_call.txt"
    text = read_file(f"{path}{earning_call_path}")
    text = strip_all_entities(strip_links(text))
    text = text.replace("[^\w\s]", "")
    text = " ".join(
        util_preprocess_text(text, stop_words=stop_words, min_word_length=4)
    )
    for word in text.split():
        if word in filter_list:
            preprocessed_list.append(word)
    preprocessed = " ".join(preprocessed_list)
    return preprocessed


def write_preprocessed_text(preprocessed_text, path):
    preprocessed_path = f"{path}/earnings_call_preprocessed.txt"
    with open(preprocessed_path, "w") as f:  # change to a+ when new file
        f.write(preprocessed_text)
    print(f"preprocessing done, writing to {path}.")


LM_SA_df = pd.read_csv("LM-SA-2020.csv")
LM_SA_dict = dict(zip(LM_SA_df.word, LM_SA_df.sentiment))
key = list(set(LM_SA_dict.values()))


def sentiment_scores(text):
    score = SentimentIntensityAnalyzer()
    vader_dict = score.polarity_scores(text)
    neg_vader = vader_dict["neg"]
    neu_vader = vader_dict["neu"]
    pos_vader = vader_dict["pos"]
    sent_TB = TextBlob(text).sentiment[0]
    return [neg_vader, neu_vader, pos_vader, sent_TB]


def LM_SA_score(text):
    sentiment_dict = {i: [] for i in key}
    count_sentiment = {i: [] for i in key}
    word_list = text.split(" ")
    for word in word_list:
        if word in LM_SA_dict.keys():
            sentiment_dict[LM_SA_dict[word]].append(word)
    for k, list in sentiment_dict.items():
        count_sentiment[k] = len(list)
    return [*count_sentiment.keys()], [*count_sentiment.values()]


if __name__ == "__main__":
    root_path = Path.cwd()
    row = []
    earnings_call_path = glob.glob(f"{root_path}/*/*/*/earnings_call.txt")
    corpus = get_corpus(earnings_call_path)
    filter_list = count_vect_filter(corpus)

    for p in meta_df.Path:
        preprocessed_text = preprocessing(p, filter_list)
        write_preprocessed_text(preprocessed_text, p)

        sentiment = sentiment_scores(preprocessed_text)
        col, LM_score = LM_SA_score(preprocessed_text)
        sentiment.extend(LM_score)
        sentiment.append(p)

        columns = ["neg_vader", "neu_vader", "pos_vader", "sent_TB"]
        columns.extend(col)
        columns.append("Path")
        row.append(sentiment)

    output_df = pd.DataFrame([r for r in row], columns=columns)
    output_df.to_csv(f"{root_path}/sentiment_df.csv")
    print(f"output csv to {root_path}/!")
