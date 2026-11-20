import json
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

own = pd.read_csv("data/ghana_own.csv")

models = {
    "public_only": joblib.load("results/model.joblib"),
    "mixed": joblib.load("results/model_mixed.joblib"),
}

results = {}
for name, m in models.items():
    pred = m.predict(own.text)
    own[f"pred_{name}"] = pred
    own[f"prob_{name}"] = m.predict_proba(own.text)[:, 1].round(3)
    results[name] = {
        "n": len(own),
        "accuracy": round(accuracy_score(own.is_scam, pred), 3),
        "precision": round(precision_score(own.is_scam, pred, zero_division=0), 3),
        "recall": round(recall_score(own.is_scam, pred, zero_division=0), 3),
        "f1": round(f1_score(own.is_scam, pred, zero_division=0), 3),
    }
print(json.dumps(results, indent=2))

with open("results/own_metrics.json", "w") as f:
    json.dump(results, f, indent=2)
own.to_csv("results/own_predictions.csv", index=False)