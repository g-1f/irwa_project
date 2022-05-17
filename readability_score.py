from asyncore import read
from readability import Readability
import pandas as pd
import numpy as np
from sentiment_analysis import read_file, get_corpus
import glob
from pathlib import Path


def readability_score(text):
    r = Readability(text)
    try:
        f_k = r.flesch_kincaid().score
        g_f = r.gunning_fog().score
        d_c = r.dale_chall().score
        spache = r.spache().score
        return [f_k, g_f, d_c, spache]
    except:
        return [0, 0, 0, 0]


if __name__ == "__main__":
    root_path = Path.cwd()
    earnings_call_path = glob.glob(f"{root_path}/*/*/*/earnings_call.txt")
    text_list = get_corpus(earnings_call_path)
    row = []
    for idx, text in enumerate(text_list):
        score = readability_score(text)
        score.append(earnings_call_path[idx])
        row.append(score)

        columns = ["f_k", "g_f", "d_c", "spache", "Path"]

    output_df = pd.DataFrame([r for r in row], columns=columns)
    output_df.Path = output_df.Path.apply(lambda x: x.strip("earnings_call.text"))
    output_df.to_csv(f"{root_path}/readability_score.csv")
