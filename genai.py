import joblib, pandas as pd

bundle = joblib.load("model.joblib")
model, FEATURES = bundle["model"], bundle["features"]

p = pd.read_csv("patients.csv")
s = pd.read_csv("sessions.csv", parse_dates=["date"])
df = s.merge(p, on="patient_id")
df["day_of_week"] = df.date.dt.dayofweek
df["transport_type"] = df["transport"]
df = pd.get_dummies(df, columns=["transport"], dtype=int)
for c in FEATURES:
    if c not in df:
        df[c] = 0

# Treat the latest date as "today" and rank patients by risk
today = df[df.date == df.date.max()].copy()
today["risk"] = model.predict_proba(today[FEATURES])[:, 1]
top = today.sort_values("risk", ascending=False).head(5)

def reasons(r):
    out = []
    if r.storm: out.append(f"storm forecast ({r.rain_in:.1f} in of rain)")
    elif r.rain_in > 0.1: out.append("rain expected")
    if r.extreme_temp: out.append("extreme temperature")
    if r.transport_type == "bus": out.append("relies on the bus")
    if r.transport_type == "medical_van": out.append("relies on a medical van")
    if r.distance_mi > 10: out.append(f"lives {r.distance_mi} miles from the clinic")
    if r.past_miss_rate >= 0.3: out.append("missed some recent sessions")
    if r.age >= 75: out.append("age 75+")
    return out or ["general risk pattern"]

def build_prompt(r, why):
    return (f"You are a dialysis clinic care coordinator. Write a short, warm text message in {r.language} "
            f"to {r.name.split()[0]} reminding them of their dialysis session tomorrow at {r.clinic}. "
            f"Their situation: {', '.join(why)}. Offer concrete help, like arranging a ride, if relevant. "
            "Under 60 words. Do not mention risk scores or medical conditions.")

def llm(prompt):
    return None  # TODO: replace with the lab's model call

def fallback(r, why):
    ride = " Reply RIDE and we'll arrange transport." if r.transport_type in ("bus", "medical_van") or r.storm else ""
    return f"Hi {r.name.split()[0]}, a reminder that your dialysis session at {r.clinic} is tomorrow. We look forward to seeing you!{ride}"

for r in top.itertuples():
    why = reasons(r)
    msg = llm(build_prompt(r, why)) or fallback(r, why)
    print(f"\n{r.name} | risk {r.risk:.0%} | {', '.join(why)}\n  -> {msg}")

