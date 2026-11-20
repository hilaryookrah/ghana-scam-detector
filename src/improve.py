import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
gh = pd.read_csv("data/ghana_test.csv")

# half of the ghana messages go to training, half stay as a held-out exam
gh_dev, gh_hold = train_test_split(gh, test_size=0.5, random_state=42, stratify=gh.is_scam)

def make_model():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

def fit_mixed(pub, ghana, weight):
    data = pd.concat([pub, ghana])
    w = np.r_[np.ones(len(pub)), np.full(len(ghana), weight)]
    m = make_model()
    m.fit(data.text, data.is_scam, clf__sample_weight=w)
    return m

def scores(model, df):
    p = model.predict(df.text)
    return {"precision": round(precision_score(df.is_scam, p, zero_division=0), 3),
            "recall": round(recall_score(df.is_scam, p), 3),
            "f1": round(f1_score(df.is_scam, p), 3)}

# choose the weight for ghana messages with cross-validation on gh_dev only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = {}
for weight in [1, 5, 10, 20]:
    fold_f1 = []
    for tr_idx, va_idx in cv.split(gh_dev.text, gh_dev.is_scam):
        m = fit_mixed(train, gh_dev.iloc[tr_idx], weight)
        fold_f1.append(scores(m, gh_dev.iloc[va_idx])["f1"])
    cv_f1[weight] = round(float(np.mean(fold_f1)), 3)
    print(f"weight {weight:>2}: cv f1 on ghana = {cv_f1[weight]}")

best_weight = max(cv_f1, key=cv_f1.get)
print("chosen weight:", best_weight)

# compare: public-only model vs mixed model, both on the same held-out ghana half
public_only = make_model().fit(train.text, train.is_scam)
mixed = fit_mixed(train, gh_dev, best_weight)

results = {
    "chosen_weight": best_weight,
    "cv_f1_by_weight": cv_f1,
    "public_only": {"public_test": scores(public_only, test),
                    "ghana_holdout": scores(public_only, gh_hold)},
    "mixed": {"public_test": scores(mixed, test),
              "ghana_holdout": scores(mixed, gh_hold)},
}
print(json.dumps(results, indent=2))

with open("results/improved_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
joblib.dump(mixed, "results/model_mixed.joblib")