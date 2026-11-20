import re
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

test = pd.read_csv("data/test.csv")
y = test.is_scam

def report(name, pred):
    print(f"{name:<18} accuracy={accuracy_score(y, pred):.3f}  "
          f"precision={precision_score(y, pred, zero_division=0):.3f}  "
          f"recall={recall_score(y, pred):.3f}  "
          f"F1={f1_score(y, pred):.3f}")

# Baseline 1: always say "genuine"
report("always genuine", [0] * len(test))

# Baseline 2: simple keyword rules
KEYWORDS = ["free", "win", "winner", "prize", "claim", "urgent", "cash",
            "congratulations", "call now", "txt", "reply", "http", "www", "£", "$"]
pattern = re.compile("|".join(re.escape(k) for k in KEYWORDS), re.I)
keyword_pred = test.text.apply(lambda t: int(bool(pattern.search(t))))
report("keyword rules", keyword_pred)