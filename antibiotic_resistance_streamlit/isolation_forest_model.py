"""
Isolation Forest Anomaly Detection Module for AMR System.
Identifies anomalous prescribing patterns, unusual consumption spikes,
and high irrationality behaviors. Produces normalized anomaly risk scores.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS_ISOLATION_FOREST = [
    "Quantity_Sold",
    "Quantity_per_1000",
    "Total_DDD",
    "Total_Monthly_DDD",
    "DDD_per_1000_per_day",
    "Antibiotic_Class_Shift_Percent",
    "Combination_Irrationality_Index_Percent",
    "Volatility_Percent",
    "Antibiotic_Share",
    "Spectrum_Weight"
]

class AMRIsolationForest:
    def __init__(self, contamination=0.08, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_columns = FEATURE_COLUMNS_ISOLATION_FOREST
        self.is_fitted = False

    def fit(self, df):
        """Fits the scaler and Isolation Forest on training dataframe."""
        X = df[self.feature_columns].copy().fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True
        return self

    def predict_anomaly_scores(self, df):
        """
        Calculates:
        - Anomaly_Flag: 1 for anomaly, 0 for normal
        - Anomaly_Raw_Score: raw decision function from IsolationForest
        - Anomaly_Risk_Score: normalized 0-1 scale (higher = more anomalous/higher risk)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting anomaly scores.")
        
        X = df[self.feature_columns].copy().fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # In sklearn IsolationForest: -1 is anomaly, 1 is normal
        raw_preds = self.model.predict(X_scaled)
        anomaly_flag = np.where(raw_preds == -1, 1, 0)
        
        # Decision function: lower values mean more anomalous
        decision_scores = self.model.decision_function(X_scaled)
        
        # Normalize score to [0, 1] range where 1 is highest anomaly severity
        # Score ranges roughly from min_val to max_val
        min_score = decision_scores.min()
        max_score = decision_scores.max()
        
        if max_score > min_score:
            # Invert so lower decision score becomes higher anomaly score
            anomaly_risk_score = (max_score - decision_scores) / (max_score - min_score)
        else:
            anomaly_risk_score = np.zeros_like(decision_scores)
            
        result_df = df.copy()
        result_df["Anomaly_Flag"] = anomaly_flag
        result_df["Anomaly_Raw_Score"] = decision_scores
        result_df["Anomaly_Risk_Score"] = np.round(anomaly_risk_score, 4)
        
        return result_df

    def save(self, filepath):
        """Saves model pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "model": self.model,
            "feature_columns": self.feature_columns,
            "contamination": self.contamination,
            "is_fitted": self.is_fitted
        }, filepath)
        print(f"Isolation Forest model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads model pipeline from disk."""
        data = joblib.load(filepath)
        instance = cls(contamination=data.get("contamination", 0.08))
        instance.scaler = data["scaler"]
        instance.model = data["model"]
        instance.feature_columns = data["feature_columns"]
        instance.is_fitted = data["is_fitted"]
        return instance

if __name__ == "__main__":
    from feature_engineering import build_complete_feature_matrix
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    sales_file = os.path.join(data_dir, "pharmacy_sales.csv")
    resistance_file = os.path.join(data_dir, "resistance_data.csv")
    
    print("Loading engineered features...")
    df, _ = build_complete_feature_matrix(sales_file, resistance_file)
    
    iso_model = AMRIsolationForest(contamination=0.08, random_state=42)
    iso_model.fit(df)
    
    scored_df = iso_model.predict_anomaly_scores(df)
    
    model_path = os.path.join(data_dir, "models", "isolation_forest.joblib")
    iso_model.save(model_path)
    
    print(f"Total records: {len(scored_df)}")
    print(f"Detected Anomalies: {scored_df['Anomaly_Flag'].sum()} ({scored_df['Anomaly_Flag'].mean()*100:.2f}%)")
    print(f"Anomaly Score Stats:\n{scored_df['Anomaly_Risk_Score'].describe()}")
