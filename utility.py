import pandas as pd
import string
import unicodedata
import contractions
import re

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np

wn = WordNetLemmatizer()
stop_words = stopwords.words("english")


def clean_word(token, stop_words, min_word_length):
    clean_token = (
        unicodedata.normalize("NFKD", token)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    clean_token = wn.lemmatize(clean_token, pos="n")
    clean_token = wn.lemmatize(clean_token, pos="v")

    if (
        (clean_token not in string.punctuation)
        and (clean_token not in stop_words)
        and (len(clean_token) >= min_word_length)
    ):

        return clean_token


def util_preprocess_text(text, stop_words=stop_words, min_word_length=3):
    if isinstance(text, str):
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub("<[^<]+?>", "", text)
        text = re.sub(r"\b\d+\b", "", text)
        tokens = word_tokenize(text)
        clean_tokens = [clean_word(w, stop_words, min_word_length) for w in tokens]
        clean_tokens = list(filter(None, clean_tokens))
        return clean_tokens


# reference https://absolutecodeworks.com/python-add-days-excluding-weekends
def add_date(startDate, daysToAdd):
    # 0 - Sunday and 6 Saturday are to be skipped
    workingDayCount = 0
    while workingDayCount < daysToAdd:
        startDate += timedelta(days=1)
        weekday = int(startDate.strftime("%w"))
        if weekday != 0 and weekday != 6:
            workingDayCount += 1

    return startDate


# reference https://stackoverflow.com/questions/8376691/how-to-remove-hashtag-user-link-of-a-tweet-using-regular-expression
def strip_links(text):
    link_regex = re.compile(
        "((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)", re.DOTALL
    )
    links = re.findall(link_regex, text)
    for link in links:
        text = text.replace(link[0], ", ")
    return text


def strip_all_entities(text):
    entity_prefixes = ["@", "#", ""]
    for separator in string.punctuation:
        if separator not in entity_prefixes:
            text = text.replace(separator, " ")
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return " ".join(words)
