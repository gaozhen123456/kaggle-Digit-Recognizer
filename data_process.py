import pandas as pd

head_split_train = 33600
tail_split_train = 8400

data_train = pd.read_csv("./data/train/train.csv").head(head_split_train)
data_train.to_csv("./data/train/data_train.csv", index=False)

data_valid = pd.read_csv("./data/train/train.csv").tail(tail_split_train)
data_valid.to_csv("./data/train/data_valid.csv",index=False)