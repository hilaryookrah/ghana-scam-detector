import json
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

model = joblib.load("results/model.joblib")
gh = pd.read_csv("data/ghana_test.csv")

gh["pred"] = model.predict(gh.text)
gh["scam_prob"] = model.predict_proba(gh.text)[:, 1].round(3)

y = gh.is_scam
tn, fp, fn, tp = confusion_matrix(y, gh.pred).ravel()
out = {
    "n_messages": len(gh),
    "n_scams": int(y.sum()),
    "accuracy": round(accuracy_score(y, gh.pred), 3),
    "precision": round(precision_score(y, gh.pred, zero_division=0), 3),
    "recall": round(recall_score(y, gh.pred), 3),
    "f1": round(f1_score(y, gh.pred), 3),
    "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
}
print(json.dumps(out, indent=2))

with open("results/ghana_metrics.json", "w") as f:
    json.dump(out, f, indent=2)
gh.to_csv("results/ghana_predictions.csv", index=False)