# Antimicrobial Resistance (AMR) Machine Learning Evaluation Report

## 1. Executive Model Performance Summary
An end-to-end Machine Learning pipeline was constructed combining **Isolation Forest Anomaly Detection**, **XGBoost Resistance Prediction**, and **Composite AMR Risk Scoring**.

### XGBoost Model Diagnostics (Out-of-Time Forward Test Split)
- **R² Determination Coefficient**: `0.9786`
- **Root Mean Squared Error (RMSE)**: `0.56%`
- **Mean Absolute Error (MAE)**: `0.39%`
- **Mean Absolute Percentage Error (MAPE)**: `1.75%`

---

## 2. Top 10 Most Influential Features (XGBoost Importance)
1. **Rolling_3M_Resistance**: `0.3822`
2. **Lag1_Resistance_Rate**: `0.2374`
3. **Lag2_Resistance_Rate**: `0.2362`
4. **Month_Number**: `0.0230`
5. **DDD_per_1000_per_day**: `0.0171`
6. **Combination_Irrationality_Index_Percent**: `0.0138`
7. **Antibiotic_Class_Shift_Percent**: `0.0105`
8. **Total_Monthly_DDD**: `0.0089`
9. **Quarter**: `0.0077`
10. **Latitude**: `0.0064`

---

## 3. Isolation Forest Anomaly Detection Summary
- **Contamination Target**: 8%
- **Anomalous Usage Patterns Identified**: 557 (8.00%)
- **Baseline Patterns**: 6403 (92.00%)

---

## 4. Multi-Factor AMR Risk Tier Breakdown
- **Critical Alert (≥ 75)**: 0 (0.0%)
- **High Risk (55–74.9)**: 0 (0.0%)
- **Moderate Risk (30–54.9)**: 1173 (16.9%)
- **Low Risk (< 30)**: 5787 (83.1%)

---

## 5. Top Highest-Risk Locations
- **Delhi**: Avg Risk Score = `36.51`, Critical Alerts = `0`
- **Kolkata**: Avg Risk Score = `34.1`, Critical Alerts = `0`
- **Mumbai**: Avg Risk Score = `33.35`, Critical Alerts = `0`
- **Bengaluru**: Avg Risk Score = `29.28`, Critical Alerts = `0`
- **Hyderabad**: Avg Risk Score = `28.35`, Critical Alerts = `0`
