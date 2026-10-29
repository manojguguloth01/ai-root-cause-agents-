# AI Root Cause and Decision Agents

## Run locally (do this on the review laptop)
    python -m venv venv
    venv\Scripts\activate        (Windows)   |   source venv/bin/activate  (Mac/Linux)
    pip install -r requirements.txt
    streamlit run app.py
Opens at http://localhost:8501

## Quick check without the UI
    python core.py

## Deploy (Streamlit Community Cloud)
1. Push this folder to a public GitHub repo (app.py, core.py, requirements.txt at repo root)
2. Go to share.streamlit.io, sign in with GitHub, New app
3. Pick the repo, branch main, main file app.py, Deploy
4. Share the URL it gives you

## Files
- core.py : data generator, anomaly detection, Root Cause Agent, Decision Agent, evaluation
- app.py  : Streamlit dashboard
- sample_incident.csv : sample data for the CSV upload option
