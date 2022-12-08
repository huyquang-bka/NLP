import pandas as pd
from util import *
from classify import classify
import pickle

path_file_test = "data/test.csv"
path_save_test = "data/test_imbalance.csv"
model_path = "model_imbalance.pkl"

with open(model_path, "rb") as f:
    clf = pickle.load(f)

df = pd.read_csv(path_file_test)
comment = df.Comment.to_list()
df["Rating"] = [classify(text) for text in tqdm(comment)]
df_revId = df["RevId"]
df_rating = df["Rating"]
new_df = pd.concat([df_revId, df_rating], axis=1)

new_df.to_csv(path_save_test, index=False, header=["RevId", "Rating"])
