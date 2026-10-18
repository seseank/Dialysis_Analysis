import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance

p = pd.read_csv("patients.csv")
s = pd.read_csv("sessions.csv", parse_dates=["date"])
df = s.merge(p, on="patient_id")
df["day_of_week"] = df.date.dt.dayofweek
df = pd.get_dummies(df, columns=["transport"], dtype=int)

FEATURES = ["age", "distance_mi", "zip_no_car_rate", "years_on_dialysis", "diabetic",
            "rain_in", "temp_max", "temp_min", "storm", "extreme_temp",
            "past_miss_rate", "day_of_week"] + [c for c in df.columns if c.startswith("transport_")]

# Train on Jan-Apr, test on May-Jun (predicting the future, like real life)
train, test = df[df.date < "2026-05-01"], df[df.date >= "2026-05-01"]
Xtr, ytr, Xte, yte = train[FEATURES], train.missed, test[FEATURES], test.missed

model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                       class_weight="balanced", random_state=0).fit(Xtr, ytr)
baseline = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)

pm, pb = model.predict_proba(Xte)[:, 1], baseline.predict_proba(Xte)[:, 1]
print(f"Test sessions: {len(test)} | actual miss rate: {yte.mean():.3f}")
print(f"Gradient boosting  AUC: {roc_auc_score(yte, pm):.3f} | AP: {average_precision_score(yte, pm):.3f}")
print(f"Logistic baseline  AUC: {roc_auc_score(yte, pb):.3f} | AP: {average_precision_score(yte, pb):.3f}")

# The pitch number: if the clinic calls only the top 10% riskiest, how many misses do they catch?
top = np.argsort(-pm)[: int(len(pm) * 0.10)]
print(f"Calling the top 10% riskiest patients catches {yte.iloc[top].sum() / yte.sum():.0%} of missed sessions")

imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
print("\nTop risk factors:")
print(pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).head(8).round(3))

joblib.dump({"model": model, "features": FEATURES}, "model.joblib")
print("\nSaved model.joblib")
