# Antibiotic Resistance Trend Predictor — Streamlit Dashboard

This is **Part 4 (Website/UI)** of the VIT Bhopal project.

## 1. Folder structure

```text
antibiotic_resistance_streamlit/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── pharmacy_sales.csv
│   ├── resistance_data.csv
│   ├── processed_features.csv
│   ├── amr_complete_predictions_and_risks.csv
│   ├── location_risk_ranking.csv
│   └── model_evaluation_report.md
├── xgboost_model.py
├── isolation_forest_model.py
├── feature_engineering.py
└── risk_scoring.py
```

## 2. Run in VS Code

Open this folder in VS Code.

Open Terminal and run:

### Mac / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Your browser should open the Streamlit dashboard.

## 3. What is already integrated

The dashboard uses the team's generated:

- `amr_complete_predictions_and_risks.csv`
- `location_risk_ranking.csv`
- engineered features
- Isolation Forest outputs
- XGBoost prediction outputs
- project risk tiers and recommended actions

The dashboard therefore matches the actual Part 1–3 data rather than using invented demo values.

## 4. CSV upload

Use the sidebar's **Upload project output CSV**.

For the complete dashboard, upload a CSV containing at least:

- `Location`
- `Month`
- `Antibiotic`
- `DDD_per_1000_per_day`
- `Combination_Irrationality_Index_Percent`
- `Volatility_Percent`
- `Predicted_Resistance_Rate`
- `Risk_Score`
- `Risk_Tier`

The supplied `amr_complete_predictions_and_risks.csv` already contains these fields.

## 5. User prediction

The User Input section uses the supplied `AMRResistanceXGBoost` implementation. It trains the project model when the prediction button is pressed and demonstrates an edited consumption profile.

This is for an **academic project demonstration only**, not a clinical decision-support system.

## 6. Important

No serialized XGBoost `.joblib` model was present in the supplied Parts 1–3 ZIP. Therefore the dashboard trains the supplied XGBoost implementation at runtime for the User Prediction section. The precomputed dashboard results remain available immediately.

If the team later provides `xgboost_resistance_model.joblib`, it can be added and the app can be changed to load that file instead of retraining.
