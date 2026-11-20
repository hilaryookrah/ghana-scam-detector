import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, confusion_matrix)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                  min_df=2, sublinear_tf=True)),
        ("clf", clf),
    ])

candidates = {
    "logistic_regression": make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced")),
    "naive_bayes": make_pipeline(MultinomialNB()),
}

# pick the model with cross-validation on train only, test stays untouched
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in candidates.items():
    scores = cross_val_score(model, train.text, train.is_scam, cv=cv, scoring="f1")
    cv_scores[name] = round(scores.mean(), 3)
    print(f"{name:<20} cv f1 = {scores.mean():.3f}")

best_name = max(cv_scores, key=cv_scores.get)
print("chosen:", best_name)

# fit on all of train, score once on test
best = candidates[best_name]
best.fit(train.text, train.is_scam)
pred = best.predict(test.text)

metrics = {
    "model": best_name,
    "cross_validated_f1": cv_scores,
    "test_accuracy": round(accuracy_score(test.is_scam, pred), 3),
    "test_precision": round(precision_score(test.is_scam, pred), 3),
    "test_recall": round(recall_score(test.is_scam, pred), 3),
    "test_f1": round(f1_score(test.is_scam, pred), 3),
}
tn, fp, fn, tp = confusion_matrix(test.is_scam, pred).ravel()
metrics["confusion"] = {"true_negative": int(tn), "false_positive": int(fp),
                        "false_negative": int(fn), "true_positive": int(tp)}
print(json.dumps(metrics, indent=2))

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
joblib.dump(best, "results/model.joblib")