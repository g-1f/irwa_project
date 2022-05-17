from turtle import forward
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re

df = pd.read_csv("meta_data.csv")
forward_pe = pd.read_csv("Forward_pe.csv")
trailing_pe = pd.read_csv("Trailing_pe.csv")
trailing_eps = pd.read_csv("Trailing_eps.csv")
market_cap = pd.read_csv("Market_cap.csv")
target = pd.read_csv("Target.csv")
meta_df = df.merge(forward_pe, how="left")
meta_df = meta_df.merge(trailing_pe, how="left")
meta_df = meta_df.merge(trailing_eps, how="left")
meta_df = meta_df.merge(market_cap, how="left")
meta_df = meta_df.merge(target, how="left")
meta_df = meta_df.loc[:, ~meta_df.columns.str.contains("^Unnamed")]

sent_df = pd.read_csv("sentiment_df.csv")
svd_df = pd.read_csv("svd_df.csv")
readability_df = pd.read_csv("readability_score.csv")

features = svd_df.merge(readability_df, how="left", on="Path")
features = features.merge(sent_df, how="left", on="Path")
features = features.loc[:, ~features.columns.str.contains("^Unnamed")]

df = meta_df.merge(features, on="Path", how="left")
df = df.drop(columns=["Earnings_url", "Path"])
# df = df[df.FY != 1]
# df = df.drop(columns=["FY"])

# CQQQ_list = []
# print(df.columns)

sector_dict = {
    "Real Estate": 1,
    "Materials": 2,
    "Consumer Discretionary": 3,
    "Consumer Staples": 4,
    "Industrials": 5,
    "Communication Services": 6,
    "Utilities": 7,
    "Financials": 8,
    "Information Technology": 9,
    "Energy": 10,
    "Health Care": 11,
}
df.Sector = df.Sector.map(sector_dict)

Q_dict = {
    "Q1": 1,
    "Q2": 2,
    "Q3": 3,
    "Q4": 4,
}
df.Quarter = df.Quarter.map(Q_dict)

year_dict = {2020: 0, 2021: 1, 2022: 2}
df.Year = df.Year.map(year_dict)


def one_hot_encoding(df, column):
    one_hot = pd.get_dummies(df[column])
    df = df.drop(columns=column)
    df = df.join(one_hot)
    return df


# df = one_hot_encoding(df, "Sector")
# df = one_hot_encoding(df, "Year")
# df = one_hot_encoding(df, "Quarter")


scaler = StandardScaler()
df[
    [
        "f_k",
        "g_f",
        "d_c",
        "spache",
        "StrongModal",
        "Positive",
        "Negative",
        "Uncertainty",
        "WeakModal",
        "Litigious",
        "Constraining",
    ]
] = scaler.fit_transform(
    df[
        [
            "f_k",
            "g_f",
            "d_c",
            "spache",
            "StrongModal",
            "Positive",
            "Negative",
            "Uncertainty",
            "WeakModal",
            "Litigious",
            "Constraining",
        ]
    ]
)

df["Target"] = df.Target * 100


def target_masking(target):
    return np.where(
        (target > -3) & (target < 3),
        0,
        np.where(target > 3, 1, np.where(target < -3, -1, 0)),
    )


df["Target_Class"] = target_masking(df.Target)
# print(df.Target)
df = df.fillna(0)
df.to_csv("dataset.csv")
