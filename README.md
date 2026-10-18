# DialytiX
**IBM Z Datathon 2026 · Theme 1: Real-Time AI for Critical Decisions (Healthcare Prioritisation)**

Missing even one dialysis session can cause fluid buildup, dangerous potassium levels and emergency room visits. Clinics usually find out about a missed session after it happens. This project predicts which patients are likely to miss their next session, using weather, transport and attendance history, so clinic staff can reach out first with a reminder or a ride.

**Demo video:** https://app.screencastify.com/watch/hJ43BZp228W19bKMqk2I
**Team:** Sejal Ankam, Nana R. Ankomah, Sachin Kumar, Theekshna Machireddy, Sahana Sendhilkumar

## How it works

1. **Data**
   - 300 synthetic patients placed in real Dallas-area ZIP codes
   - 112 real dialysis clinics from the CMS Dialysis Facility dataset
   - Real car-access rates by ZIP from the U.S. Census (ACS table B25044)
   - 6 months of real daily weather from NOAA (DFW Airport station)
   - About 23,000 simulated dialysis sessions, each marked attended or missed

2. **Predictive AI (scikit-learn)**
   - Predicts each patient's chance of missing their next session
   - Trained on January to April, tested on May to June, so it is tested on future dates it has never seen
   - Calling the top 10% riskiest patients catches 32% of missed sessions, about 3x better than random
   - Top risk factors: rainfall, relying on the bus, age

3. **Generative AI (IBM Granite via llama.cpp)**
   - Granite 4.0 runs locally on LinuxONE and writes a short outreach message for each high-risk patient, in the patient's language
   - Patient data never leaves the server

4. **Dashboard (Flask)**
   - Lists today's highest-risk patients, explains why each one was flagged, and generates a Granite message with one click

Model:[https://drive.google.com/file/d/1IlnhGp3qkjx5YPOQ2PI9KKXK7nr2QoPB/view?usp=sharing](url)

## Why a patient gets flagged

| Factor | Cutoff |
|---|---|
| Storm | 0.5 in of rain or more |
| Rain | more than 0.1 in |
| Extreme temperature | high of 95°F or more, or low of 32°F or less |
| Recent misses | 2 or more of the last 6 sessions |
| Transport | bus or medical van |
| Distance | more than 10 miles from the clinic |
| Age | 75 or older |

## Running on IBM LinuxONE

Everything runs on the LinuxONE Community Cloud (s390x) inside the lab's Jupyter Lab container, which includes llama.cpp built for the on-chip AI accelerator (zDNN).

![Dashboard](screenshots/dashboard.png)
![Container running](screenshots/docker.png)
![Model results](screenshots/model.png)

## Files

| File | What it does |
|---|---|
| `generate_patients.py` | Builds patients from real CMS clinic and Census data |
| `generate_session.py` | Simulates 6 months of sessions using real NOAA weather |
| `train_model.py` | Trains and tests the risk model, saves `model.joblib` |
| `genai.py` | Ranks today's patients and writes Granite messages |
| `app.py` | Flask dashboard |

## How to run

    docker exec -it notebook bash -c "cd /home/jovyan/project && python train_model.py && python app.py"

Then open port 5000. The CSV files are already included. Rebuilding `patients.csv` needs a free Census API key.

## Limitations and next steps

- Patients and attendance are synthetic. Clinic locations, car access and weather are real. The next step would be testing on real clinic attendance records.
- Granite 4.0 350M is a small model, so messages can be repetitive. A larger Granite model would write better messages.
- Real deployment would add live weather forecasts, transit delay data and SMS sending.

