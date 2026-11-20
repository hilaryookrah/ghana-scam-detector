import pandas as pd

df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "text"])
print(df.shape)
print(df.label.value_counts())
print(df.label.value_counts(normalize=True).round(3))
print("duplicate rows:", df.duplicated().sum())
print(df.text.str.len().groupby(df.label).describe().round(1))