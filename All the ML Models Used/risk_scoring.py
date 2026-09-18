"""
Composite AMR Risk Scoring Engine.
Synthesizes Resistance Rates, Isolation Forest Anomaly Scores,
Combination Irrationality Index, and Spectrum Shift Dynamics into a 0-100 Risk Index.
"""

import pandas as pd
import numpy as np

# Risk Component Weights (Sum to 1.0)
WEIGHT_RESISTANCE = 0.40
WEIGHT_ANOMALY = 0.25
WEIGHT_IRRATIONALITY = 0.20
WEIGHT_SPECTRUM_SHIFT = 0.15

# Spectrum mapping for risk weighting
SPECTRUM_FACTOR = {
    "Narrow": 30.0,
    "Broad": 70.0,
    "Very Broad": 100.0
}

def calculate_composite_risk_score(df, use_predicted_resistance=True):
    """
    Computes a multi-criteria 0-100 AMR Risk Score:
    - Resistance Rate Pressure (40%)
    - Isolation Forest Anomaly Score (25%)
    - Combination Irrationality (20%)
    - Spectrum & Class Shift Dynamics (15%)
    """
    result_df = df.copy()
    
    # 1. Resistance component
    if use_predicted_resistance and "Predicted_Resistance_Rate" in result_df.columns:
        res_col = "Predicted_Resistance_Rate"
    elif "Resistance_Rate" in result_df.columns:
        res_col = "Resistance_Rate"
    else:
        raise ValueError("Neither Resistance_Rate nor Predicted_Resistance_Rate found.")
        
    res_score = np.clip(result_df[res_col].fillna(0), 0, 100)
    
    # 2. Anomaly component (Isolation Forest 0-1 score -> 0-100)
    if "Anomaly_Risk_Score" in result_df.columns:
        anomaly_score = np.clip(result_df["Anomaly_Risk_Score"].fillna(0) * 100.0, 0, 100)
    else:
        anomaly_score = pd.Series(0, index=result_df.index)
        
    # 3. Combination Irrationality component
    if "Combination_Irrationality_Index_Percent" in result_df.columns:
        irrat_score = np.clip(result_df["Combination_Irrationality_Index_Percent"].fillna(0), 0, 100)
    else:
        irrat_score = pd.Series(0, index=result_df.index)
        
    # 4. Spectrum & Class Shift component
    spectrum_base = result_df["Spectrum"].map(SPECTRUM_FACTOR).fillna(50.0) if "Spectrum" in result_df.columns else 50.0
    shift_val = np.clip(result_df.get("Antibiotic_Class_Shift_Percent", 0).fillna(0), 0, 100)
    spectrum_shift_score = (spectrum_base * 0.6) + (shift_val * 0.4)
    
    # Weighted composite score
    composite_score = (
        (res_score * WEIGHT_RESISTANCE) +
        (anomaly_score * WEIGHT_ANOMALY) +
        (irrat_score * WEIGHT_IRRATIONALITY) +
        (spectrum_shift_score * WEIGHT_SPECTRUM_SHIFT)
    )
    
    result_df["Risk_Score"] = np.round(composite_score, 2)
    
    # Assign Clinical / Stewardship Risk Tier
    conditions = [
        result_df["Risk_Score"] < 30,
        (result_df["Risk_Score"] >= 30) & (result_df["Risk_Score"] < 55),
        (result_df["Risk_Score"] >= 55) & (result_df["Risk_Score"] < 75),
        result_df["Risk_Score"] >= 75
    ]
    tiers = ["Low Risk", "Moderate Risk", "High Risk", "Critical Alert"]
    
    result_df["Risk_Tier"] = np.select(conditions, tiers, default="Moderate Risk")
    
    # Action recommendations
    action_map = {
        "Low Risk": "Standard surveillance; maintain current stewardship guidelines.",
        "Moderate Risk": "Elevated vigilance; audit dispensing patterns for reserve classes.",
        "High Risk": "Targeted stewardship review; investigate combination & high-volume prescriptions.",
        "Critical Alert": "Urgent intervention required; audit microbiology isolates and restrict empirical reserve use."
    }
    result_df["Recommended_Action"] = result_df["Risk_Tier"].map(action_map)
    
    return result_df

def generate_location_risk_summary(df):
    """Aggregates risk metrics by Location and Month for executive overview."""
    summary = (
        df.groupby(["Location", "Month"])
        .agg(
            Avg_Risk_Score=("Risk_Score", "mean"),
            Max_Risk_Score=("Risk_Score", "max"),
            High_Risk_Drugs_Count=("Risk_Tier", lambda x: (x.isin(["High Risk", "Critical Alert"])).sum()),
            Avg_Predicted_Resistance=("Predicted_Resistance_Rate", "mean") if "Predicted_Resistance_Rate" in df.columns else ("Resistance_Rate", "mean"),
            Total_Monthly_Quantity=("Quantity_Sold", "sum"),
            Avg_Anomaly_Score=("Anomaly_Risk_Score", "mean") if "Anomaly_Risk_Score" in df.columns else ("Risk_Score", lambda x: 0)
        )
        .reset_index()
    )
    summary["Avg_Risk_Score"] = np.round(summary["Avg_Risk_Score"], 2)
    summary["Max_Risk_Score"] = np.round(summary["Max_Risk_Score"], 2)
    return summary.sort_values("Avg_Risk_Score", ascending=False).reset_index(drop=True)
