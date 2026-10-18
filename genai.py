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

from llama_cpp import Llama
MODEL_PATH = "/home/ibm-user/granite-4.0-h-350m-base-be.f16.gguf"
_granite = Llama(model_path=MODEL_PATH, n_ctx=2048, verbose=False)

def build_prompt(r, why):
    return (
        "Below are short reminder texts a dialysis clinic sends to patients.\n\n"
        "###\nPatient: Maria, Spanish speaker. Situation: rain expected, relies on the bus.\n"
        "Message: Hola Maria, le recordamos su sesión de diálisis de mañana. Si la lluvia complica el autobús, "
        "responda RIDE y le enviaremos transporte. ¡Le esperamos!\n\n"
        "###\nPatient: James, English speaker. Situation: lives 14 miles from the clinic, age 75+.\n"
        "Message: Hi James, just a reminder about your dialysis session tomorrow. It's a long trip, "
        "so reply RIDE if you'd like us to arrange transport. See you soon!\n\n"
         "###\nPatient: Ana, English speaker. Situation: extreme temperature, relies on the bus.\n"
        "Message: Hi Ana, a reminder about your dialysis session tomorrow. With the extreme heat, waiting "
        "at the bus stop can be risky, so reply RIDE and we'll send transport. Stay cool!\n\n"
        f"###\nPatient: {r.name.split()[0]}, {r.language} speaker. Situation: {', '.join(why)}.\n"
        "Message:"
    )

def llm(prompt):
    out = _granite(prompt, max_tokens=90, temperature=0.3, stop=["###", "\n\n"])
    return out["choices"][0]["text"].strip() or None


def fallback(r, why):
    ride = " Reply RIDE and we'll arrange transport." if r.transport_type in ("bus", "medical_van") or r.storm else ""
    return f"Hi {r.name.split()[0]}, a reminder that your dialysis session at {r.clinic} is tomorrow. We look forward to seeing you!{ride}"

if __name__ == "__main__":
    for r in top.itertuples():
        why = reasons(r)
        msg = llm(build_prompt(r, why)) or fallback(r, why)
        print(f"\n{r.name} | risk {r.risk:.0%} | {', '.join(why)}\n  -> {msg}")

