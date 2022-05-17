from asyncio import FastChildWatcher
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta
import pandas as pd
from multiprocessing import Pool
from multiprocessing import Process
from pandas_datareader import data
import numpy as np
from utility import add_date

df = pd.read_csv("meta_data.csv")


def get_forward_pe(df):
    l = []
    for ticker in df.Ticker:
        try:
            forward_pe = float(
                data.get_quote_yahoo(ticker)["forwardPE"].apply(
                    lambda x: str(x).split(" ")
                )[-1][0]
            )
            l.append(forward_pe)
        except:
            l.append(0)
    df["Forward_pe"] = l
    df.to_csv("Forward_pe.csv")


def get_trailing_pe(df):
    l = []
    for ticker in df.Ticker:
        try:
            trailing_pe = float(
                data.get_quote_yahoo(ticker)["trailingPE"].apply(
                    lambda x: str(x).split(" ")
                )[-1][0]
            )
            l.append(trailing_pe)
        except:
            l.append(0)
    df["Trailing_pe"] = l
    df.to_csv("Trailing_pe.csv")


def get_traling_eps(df):
    l = []
    for ticker in df.Ticker:
        try:
            trailing_eps = float(
                data.get_quote_yahoo(ticker)["epsTrailingTwelveMonths"].apply(
                    lambda x: str(x).split(" ")
                )[-1][0]
            )
            l.append(trailing_eps)
        except:
            l.append(0)
    df["Trailing_eps"] = l
    df.to_csv("Trailing_eps.csv")


def get_marketcap(df):
    l = []
    for ticker in df.Ticker:
        try:
            cap = data.get_quote_yahoo(ticker)["marketCap"].values[0]
            l.append(cap)
        except:
            l.append(np.NAN)
    df["Market_cap"] = l
    df.to_csv("Market_cap.csv")


def target(ticker, date):
    try:
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = add_date(start_date, 2)  # return over 1 week horizon

        df = si.get_data(
            ticker, start_date=start_date, end_date=end_date, interval="1d"
        )
        df.reset_index(inplace=True)
        target = (df.iloc[-1].close - df.iloc[0].close) / df.iloc[0].close
        return target
    except:
        return -1


def get_target(df):
    t = []
    count = 0
    for ticker, date in zip(df.Ticker, df.Date):
        return_ = target(ticker, date)
        t.append(return_)
        count += 1
        print(count)
    df["Target"] = t
    df.to_csv("Target.csv")


if __name__ == "__main__":
    p1 = Process(target=get_marketcap, args=(df,))
    p1.start()
    p1.join()
    p2 = Process(target=get_forward_pe, args=(df,))
    p2.start()
    p2.join()
    p3 = Process(target=get_trailing_pe, args=(df,))
    p3.start()
    p3.join()
    p4 = Process(target=get_traling_eps, args=(df,))
    p4.start()
    p4.join()
    p = Process(target=get_target, args=(df,))
    p.start()
    p.join()
