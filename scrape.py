# import time
import requests
from bs4 import BeautifulSoup as bs4
import re
from pathlib import Path
import os
import pandas as pd
import yahoo_fin.stock_info as si
from datetime import datetime, timedelta

root_path = Path.cwd()
base_url = "https://www.fool.com"
NDX_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
SP_500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def get_composite_set(NDX_url, SP_url):
    response = requests.get(NDX_url)
    soup = bs4(response.text, "html.parser")
    table = soup.find_all("table")
    df = pd.read_html(str(table))
    NQ_100_ticker_list = df[3]["Ticker"].to_list()
    NQ_100_sector_list = df[3]["GICS Sector"].to_list()
    ndx_dict = dict(zip(NQ_100_ticker_list, NQ_100_sector_list))

    response = requests.get(SP_url)
    soup = bs4(response.text, "html.parser")
    table = soup.find_all("table")
    df = pd.read_html(str(table))
    SP_500_ticker_list = df[0].Symbol.to_list()
    SP_500_sector_list = df[0]["GICS Sector"].to_list()
    sp_dict = dict(zip(SP_500_ticker_list, SP_500_sector_list))
    composite_dict = {**ndx_dict, **sp_dict}
    return composite_dict


# def get_eps_stats(ticker, date):
#     for i in si.get_earnings_history(ticker):
#         if i["startdatetime"][:10] == date:
#             return i["epsestimate"], i["epsactual"], i["epssurprisepct"]


# def get_PE(ticker):
#     Trailing_PE = si.get_stats_valuation(ticker).iloc[2, 1]
#     Forward_PE = si.get_stats_valuation(ticker).iloc[3, 1]
#     return Forward_PE, Trailing_PE


# def target1(ticker, date):
#     start_date = datetime.strptime(date, "%Y-%m-%d")
#     end_date = start_date + timedelta(days=8)  # return over 1 week horizon
#     df = si.get_data(ticker, start_date=start_date, end_date=end_date, interval="1d")
#     df.reset_index(inplace=True)
#     target = (df.iloc[-1].close - df.iloc[0].close) / df.iloc[0].close
#     return target


# def target2(ticker, date):
#     start_date = datetime.strptime(date, "%Y-%m-%d")
#     end_date = start_date + timedelta(days=2)  # return of 1 day after earning call
#     df = si.get_data(ticker, start_date=start_date, end_date=end_date, interval="1d")

#     df.reset_index(inplace=True)
#     target = (df.iloc[-1].close - df.iloc[0].close) / df.iloc[0].close
#     return target


def get_metadata(pages, target_dict):
    row = []
    d = {}
    url_list = []

    for i in range(1, pages):
        print(f"Searching page {i}...")
        url = f"https://www.fool.com/earnings-call-transcripts/?page={i}"
        response = requests.get(url)
        page = response.text
        soup = bs4(page, "html.parser")
        for item in soup.find_all(attrs={"data-id": True}):
            try:
                title = item["title"]
                ticker = re.search(r"\((.*)\)", item["title"]).group(1)
                year = title.split(" ")[-4]
                if ticker in target_dict.keys():
                    sector = target_dict[ticker]
                    href = item["href"]
                    date_list = href.split("/")[-5:-2]
                    date = "-".join(date_list)
                    year = title.split(" ")[-4]
                    quarter = title.split(" ")[-5]
                    earnings_url = base_url + item["href"]
                    path = f"{root_path}/{ticker}/{year}/{quarter}/"
                    # eps_estimate, eps_actual, eps_suprise = get_eps_stats(ticker, date)
                    # foward_pe, trailing_pe = get_PE(ticker)
                    # target1 = target1(ticker, date)
                    # target2 = target2(ticker, date)

                    row.append(
                        [
                            ticker,
                            sector,
                            date,
                            year,
                            quarter,
                            # eps_estimate,
                            # eps_actual,
                            # eps_suprise,
                            earnings_url,
                            # foward_pe,
                            # trailing_pe,
                            str(path),
                            # target1,
                            # target2,
                        ]
                    )
                    d[earnings_url] = path
                    url_list.append(earnings_url)
                    print(f"generated {ticker}: {year} {quarter} metadata.")

            except:
                continue

            meta_df = pd.DataFrame(
                [r for r in row],
                columns=[
                    "Ticker",
                    "Sector",
                    "Date",
                    "Year",
                    "Quarter",
                    # "Eps_estimate",
                    # "Eps_actual",
                    # "Eps_suprise",
                    "Earnings_url",
                    "Path",
                ],
            )
            meta_df.to_csv(f"{root_path}/meta_data.csv")
    return url_list, d


def file_is_empty(path):
    return os.stat(path).st_size == 0


def get_transcript(url, d):
    response = requests.get(url)
    soup = bs4(response.text, "html.parser")
    mark = soup.select_one("h2:nth-of-type(2)")
    for p in mark.nextSiblingGenerator():
        if p.name == "h2":
            break
        else:
            if hasattr(p, "text"):
                transcript = p.text
                transcript_path = f"{d[url]}earnings_call.txt"
                if not os.path.exists(d[url]):
                    os.makedirs(d[url])
                with open(transcript_path, "a+") as f:
                    f.write(transcript)
    print(f"Written transcript to {d[url]}")


if __name__ == "__main__":
    composite_dict = get_composite_set(NDX_url, SP_500_url)
    pages = 500
    url_list, d = get_metadata(pages, composite_dict)
    for url in url_list:
        get_transcript(url, d)
