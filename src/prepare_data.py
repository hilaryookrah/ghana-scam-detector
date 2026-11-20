import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/sms.tsv", sep="\t", header=None, names=["label", "text"])
before = len(df)
df = df.drop_duplicates()
print(f"removed {before - len(df)} duplicates, {len(df)} left")

df["is_scam"] = (df.label == "spam").astype(int)

# stratify so train and test keep the same scam share
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.is_scam)
train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)
print("train:", len(train), "scam share:", round(train.is_scam.mean(), 3))
print("test:", len(test), "scam share:", round(test.is_scam.mean(), 3))