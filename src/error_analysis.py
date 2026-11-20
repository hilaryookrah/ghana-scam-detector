import joblib
import pandas as pd

model = joblib.load("results/model.joblib")
test = pd.read_csv("data/test.csv")

test["pred"] = model.predict(test.text)
test["scam_prob"] = model.predict_proba(test.text)[:, 1].round(3)

missed = test[(test.is_scam == 1) & (test.pred == 0)]
false_alarms = test[(test.is_scam == 0) & (test.pred == 1)]

pd.set_option("display.max_colwidth", 140)
pd.set_option("display.width", 200)

print(f"missed scams: {len(missed)}")
print(missed[["scam_prob", "text"]].to_string(index=False))
print()
print(f"false alarms: {len(false_alarms)}")
print(false_alarms[["scam_prob", "text"]].to_string(index=False))

pd.concat([missed.assign(kind="missed"), false_alarms.assign(kind="false_alarm")]) \
  .to_csv("results/errors.csv", index=False)