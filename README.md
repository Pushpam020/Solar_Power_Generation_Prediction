# 🔆 Solar Power Generation Prediction – Streamlit App

## Files
- `app.py` – Streamlit UI (loads scaler + model and predicts)
- `requirements.txt` – Python dependencies
- `best_model.pkl` – (add your trained best model here)
- `scaler.pkl` – (add your fitted StandardScaler here)

## How to run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push these files to a **public GitHub repo**.
2. Go to https://share.streamlit.io → "New app" → choose repo → set file to `app.py`.
3. Click **Deploy**. Make sure `best_model.pkl` and `scaler.pkl` are committed to the repo.
