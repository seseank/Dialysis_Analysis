import io, numpy as np, pandas as pd, requests, pgeocode
from faker import Faker

rng = np.random.default_rng(42)
fake = Faker(); Faker.seed(42)
N = 300
DALLAS = ("750", "751", "752", "753")
nomi = pgeocode.Nominatim("us")

# 1. Real dialysis clinics (CMS)
url = "https://data.cms.gov/provider-data/sites/default/files/resources/c04d84bc5c641284494bee4f20f17f9c_1781625941/DFC_FACILITY.csv"
fac = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text), dtype=str)
col = lambda key: next(c for c in fac.columns if key in c.lower())
fac["zip"] = fac[col("zip")].str[:5]
fac = fac[fac["zip"].str.startswith(DALLAS)].copy()
fac["clinic"] = fac[col("facility name")]
g = nomi.query_postal_code(fac["zip"].tolist())
fac["lat"], fac["lon"] = g["latitude"].values, g["longitude"].values
fac = fac.dropna(subset=["lat"]).reset_index(drop=True)

# 2. Car access by ZIP (Census ACS)
import os; acs = requests.get("https://api.census.gov/data/2022/acs/acs5?get=B25044_001E,B25044_003E,B25044_010E&for=zip%20code%20tabulation%20area:*&key=" + os.environ["CENSUS_KEY"], timeout=120).json()
 
acs = pd.DataFrame(acs[1:], columns=["households", "own_nocar", "rent_nocar", "zip"])
acs = acs[acs["zip"].str.startswith(DALLAS)]
acs[["households", "own_nocar", "rent_nocar"]] = acs[["households", "own_nocar", "rent_nocar"]].astype(int).clip(lower=0)
acs = acs[acs.households > 0]
acs["no_car_rate"] = (acs.own_nocar + acs.rent_nocar) / acs.households

# 3. Place patients in real ZIPs
p = acs.sample(N, replace=True, weights="households", random_state=42).reset_index(drop=True)
g = nomi.query_postal_code(p["zip"].tolist())
p["lat"] = g["latitude"].values + rng.normal(0, 0.01, N)
p["lon"] = g["longitude"].values + rng.normal(0, 0.01, N)
p = p.dropna(subset=["lat"]).reset_index(drop=True)
n = len(p)

# 4. Distance to nearest real clinic (x1.3 for road distance)
la1, lo1 = np.radians(p.lat.values)[:, None], np.radians(p.lon.values)[:, None]
la2, lo2 = np.radians(fac.lat.values)[None, :], np.radians(fac.lon.values)[None, :]
d = 3958.8 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
p["distance_mi"] = (d.min(1) * 1.3).round(1)
p["clinic"] = fac["clinic"].values[d.argmin(1)]

# 5. Transport based on the ZIP's no-car rate
no_car = rng.random(n) < np.clip(p.no_car_rate.values * 4 + 0.15, 0, 0.9)
p["transport"] = np.where(no_car,
                          rng.choice(["bus", "medical_van", "family"], n, p=[.45, .35, .2]),
                          rng.choice(["own_car", "family"], n, p=[.8, .2]))

patients = pd.DataFrame({
    "patient_id": range(1, n + 1),
    "name": [fake.name() for _ in range(n)],
    "age": rng.integers(30, 90, n),
    "zip": p["zip"],
    "lat": p["lat"].round(4),
    "lon": p["lon"].round(4),
    "clinic": p["clinic"],
    "distance_mi": p["distance_mi"],
    "zip_no_car_rate": p["no_car_rate"].round(3),
    "transport": p["transport"],
    "schedule": rng.choice(["MWF", "TTS"], n),
    "years_on_dialysis": rng.gamma(2, 2, n).round(1),
    "diabetic": rng.choice([0, 1], n, p=[.55, .45]),
    "language": rng.choice(["English", "Spanish", "Tamil", "Vietnamese"], n, p=[.65, .2, .08, .07]),
})

patients.to_csv("patients.csv", index=False)
print(len(fac), "real Dallas-area clinics found")
print(patients.transport.value_counts())
print(patients.head())

