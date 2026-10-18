import numpy as np, pandas as pd

rng = np.random.default_rng(7)
patients = pd.read_csv("patients.csv")
weather = pd.read_csv("weather.csv")

weather["date"] = pd.to_datetime(weather["DATE"])
weather["rain_in"] = weather["PRCP"].fillna(0)
weather["temp_max"] = weather["TMAX"].interpolate().bfill()
weather["temp_min"] = weather["TMIN"].interpolate().bfill()
weather["storm"] = (weather.rain_in >= 0.5).astype(int)
weather["extreme_temp"] = ((weather.temp_max >= 95) | (weather.temp_min <= 32)).astype(int)
weather = weather[["date", "rain_in", "temp_max", "temp_min", "storm", "extreme_temp"]]

days = {"MWF": [0, 2, 4], "TTS": [1, 3, 5]}
transport_risk = {"own_car": 0, "family": .3, "medical_van": .5, "bus": .9}
habit = rng.normal(0, 1, len(patients))  # hidden per-patient tendency to miss

rows = []
for i, p in enumerate(patients.itertuples()):
    for w in weather[weather.date.dt.dayofweek.isin(days[p.schedule])].itertuples():
        logit = (-3.4 + 0.6 * habit[i] + 0.04 * p.distance_mi
                 + transport_risk[p.transport] + 1.1 * w.storm + 0.6 * w.rain_in
                 + 0.8 * w.storm * (p.transport == "bus")
                 + 0.7 * w.extreme_temp + 0.3 * (p.age >= 75) + 0.2 * p.diabetic)
        missed = int(rng.random() < 1 / (1 + np.exp(-logit)))
        rows.append([p.patient_id, w.date, w.rain_in, w.temp_max, w.temp_min,
                     w.storm, w.extreme_temp, missed])

s = pd.DataFrame(rows, columns=["patient_id", "date", "rain_in", "temp_max", "temp_min",
                                "storm", "extreme_temp", "missed"])
s = s.sort_values(["patient_id", "date"])
s["past_miss_rate"] = (s.groupby("patient_id")["missed"]
                       .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())).fillna(0)

s.to_csv("sessions.csv", index=False)
print(len(s), "sessions | miss rate:", round(s.missed.mean(), 3))
print("miss rate on storm days:", round(s[s.storm == 1].missed.mean(), 3))

