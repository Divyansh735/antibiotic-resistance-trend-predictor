"""
XGBoost Resistance Prediction Module for AMR System.
Predicts antibiotic resistance rates using consumption dynamics,
temporal lag patterns, and Isolation Forest anomaly signals.
"""

import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS_XGBOOST = [
    "Population",
    "Latitude",
    "Longitude",
    "Quantity_Sold",
    "Total_DDD",
    "Quantity_per_1000",
    "Total_Monthly_DDD",
    "DDD_per_1000_per_day",
    "Antibiotic_Share",
    "Spectrum_Weight",
    "Is_Combination",
    "Antibiotic_Class_Shift_Percent",
    "Combination_Irrationality_Index_Percent",
    "Volatility_Percent",
    "Lag1_Quantity_Sold",
    "Lag2_Quantity_Sold",
    "Lag1_Resistance_Rate",
    "Lag2_Resistance_Rate",
    "Rolling_3M_Quantity",
    "Rolling_3M_Resistance",
    "Month_Number",
    "Quarter",
    "Anomaly_Flag",
    "Anomaly_Risk_Score"
]

CATEGORICAL_COLUMNS = ["Antibiotic", "Class", "Spectrum"]

class AMRResistanceXGBoost:
    def __init__(self, n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.encoders = {}
        self.feature_columns = FEATURE_COLUMNS_XGBOOST
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.trained_features = []

    def _prepare_data(self, df, fit_encoders=False):
        """Encodes categorical variables and extracts feature matrix."""
        df_encoded = df.copy()
        
        for col in CATEGORICAL_COLUMNS:
            if col in df_encoded.columns:
                if fit_encoders:
                    le = LabelEncoder()
                    df_encoded[col + "_Code"] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders.get(col)
                    if le is not None:
                        # Handle unseen categories gracefully
                        classes = dict(zip(le.classes_, range(len(le.classes_))))
                        df_encoded[col + "_Code"] = df_encoded[col].astype(str).map(classes).fillna(-1).astype(int)
                    else:
                        df_encoded[col + "_Code"] = 0
                        
        encoded_feature_cols = self.feature_columns + [c + "_Code" for c in CATEGORICAL_COLUMNS if c in df_encoded.columns]
        X = df_encoded[encoded_feature_cols].copy().fillna(0)
        return X, encoded_feature_cols

    def train_and_evaluate_temporal_split(self, df, test_size=0.2):
        """
        Splits dataset temporally (chronological split) to simulate realistic forward forecasting,
        trains the XGBoost model, and calculates comprehensive performance metrics.
        """
        df_sorted = df.sort_values("Month").reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_idx].copy()
        test_df = df_sorted.iloc[split_idx:].copy()
        
        X_train, self.trained_features = self._prepare_data(train_df, fit_encoders=True)
        y_train = train_df["Resistance_Rate"].values
        
        X_test, _ = self._prepare_data(test_df, fit_encoders=False)
        y_test = test_df["Resistance_Rate"].values
        
        print(f"Training XGBoost model on {len(X_train)} samples, testing on {len(X_test)} samples...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        self.is_fitted = True
        
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        metrics = {
            "train_r2": float(r2_score(y_train, train_preds)),
            "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_preds))),
            "train_mae": float(mean_absolute_error(y_train, train_preds)),
            "test_r2": float(r2_score(y_test, test_preds)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
            "test_mae": float(mean_absolute_error(y_test, test_preds)),
            "test_mape": float(np.mean(np.abs((y_test - test_preds) / np.clip(y_test, 1e-5, None))) * 100)
        }
        
        test_df["Predicted_Resistance_Rate"] = np.round(test_preds, 2)
        test_df["Resistance_Error"] = np.round(test_preds - y_test, 2)
        test_df["Resistance_Abs_Error"] = np.round(np.abs(test_preds - y_test), 2)
        
        return metrics, train_df, test_df

    def predict(self, df):
        """Predicts resistance rate on input dataframe."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting.")
        X, _ = self._prepare_data(df, fit_encoders=False)
        preds = self.model.predict(X)
        result_df = df.copy()
        result_df["Predicted_Resistance_Rate"] = np.round(preds, 2)
        return result_df

    def get_feature_importances(self):
        """Returns feature importance dataframe sorted descending."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importances.")
        importances = self.model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": self.trained_features,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return fi_df

    def save(self, filepath):
        """Saves XGBoost pipeline to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "encoders": self.encoders,
            "feature_columns": self.feature_columns,
            "trained_features": self.trained_features,
            "is_fitted": self.is_fitted
        }, filepath)
        print(f"XGBoost Resistance model successfully saved to: {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads XGBoost pipeline from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.encoders = data["encoders"]
        instance.feature_columns = data["feature_columns"]
        instance.trained_features = data["trained_features"]
        instance.is_fitted = data["is_fitted"]
        return instance

if __name__ == "__main__":
    from feature_engineering import build_complete_feature_matrix
    from isolation_forest_model import AMRIsolationForest
    
    data_dir = os.path.dirname(os.path.abspath(__file__))
    sales_file = os.path.join(data_dir, "pharmacy_sales.csv")
    resistance_file = os.path.join(data_dir, "resistance_data.csv")
    
    print("Loading data & generating features...")
    df, _ = build_complete_feature_matrix(sales_file, resistance_file)
    
    print("Running Isolation Forest for anomaly signals...")
    iso_model = AMRIsolationForest()
    iso_model.fit(df)
    scored_df = iso_model.predict_anomaly_scores(df)
    
    print("Training XGBoost Regressor...")
    xgb_pipeline = AMRResistanceXGBoost()
    metrics, train_df, test_df = xgb_pipeline.train_and_evaluate_temporal_split(scored_df)
    
    print("\n--- XGBoost Performance Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
        
    print("\nTop 10 Feature Importances:")
    print(xgb_pipeline.get_feature_importances().head(10))
    
    model_path = os.path.join(data_dir, "models", "xgboost_resistance_model.joblib")
    xgb_pipeline.save(model_path)
