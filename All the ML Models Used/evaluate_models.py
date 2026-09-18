"""
Comprehensive Model Evaluation and Diagnostic Suite for AMR ML System.
Produces performance metrics, residual diagnostics, feature importance charts,
anomaly distribution analyses, and risk tier summaries.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["figure.autolayout"] = True

def compute_regression_metrics(y_true, y_pred):
    """Calculates R2, RMSE, MAE, and MAPE."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-5, None))) * 100
    return {
        "R2_Score": round(float(r2), 4),
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MAPE_Percent": round(float(mape), 2)
    }

def plot_regression_diagnostics(test_df, out_dir):
    """Plots Predicted vs Actual and Residual analysis."""
    y_true = test_df["Resistance_Rate"].values
    y_pred = test_df["Predicted_Resistance_Rate"].values
    residuals = y_pred - y_true
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    
    # 1. Actual vs Predicted Scatter
    ax1 = axes[0]
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color="#1f77b4", edgecolor="w", s=40, ax=ax1)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", lw=2, label="Ideal Fit (y=x)")
    
    metrics = compute_regression_metrics(y_true, y_pred)
    metrics_text = f"R² = {metrics['R2_Score']:.4f}\nRMSE = {metrics['RMSE']:.2f}%\nMAE = {metrics['MAE']:.2f}%"
    ax1.text(0.05, 0.90, metrics_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))
    
    ax1.set_title("XGBoost: Predicted vs Actual Resistance Rate", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Actual Resistance Rate (%)", fontsize=11)
    ax1.set_ylabel("Predicted Resistance Rate (%)", fontsize=11)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Residual Distribution
    ax2 = axes[1]
    sns.histplot(residuals, kde=True, color="#2ca02c", bins=30, ax=ax2)
    ax2.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    ax2.set_title("Prediction Residual Distribution (Errors)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Residual (Predicted - Actual %)", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(out_dir, "xgboost_regression_diagnostics.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved diagnostic plot: {plot_path}")
    return plot_path

def plot_feature_importances(fi_df, out_dir, top_n=15):
    """Plots top feature importances from XGBoost."""
    plt.figure(figsize=(10, 7), dpi=300)
    top_fi = fi_df.head(top_n).sort_values("Importance", ascending=True)
    
    bars = plt.barh(top_fi["Feature"], top_fi["Importance"], color="#3b528b", edgecolor="none", height=0.65)
    plt.title(f"Top {top_n} Most Influential Features (XGBoost)", fontsize=13, fontweight="bold")
    plt.xlabel("Relative Importance (Gain / Weight)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6, axis="x")
    
    plot_path = os.path.join(out_dir, "feature_importance_ranking.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved feature importance plot: {plot_path}")
    return plot_path

def plot_isolation_forest_analysis(df, out_dir):
    """Plots anomaly score distribution and resistance rate by anomaly flag."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. Anomaly Risk Score distribution
    ax1 = axes[0]
    sns.histplot(df["Anomaly_Risk_Score"], bins=35, kde=True, color="#d62728", ax=ax1)
    ax1.axvline(df["Anomaly_Risk_Score"].quantile(0.92), color="black", linestyle="--", label="92nd Percentile Threshold")
    ax1.set_title("Isolation Forest: Anomaly Risk Score Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Anomaly Risk Score (0 = Normal, 1 = Severe Anomaly)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # 2. Resistance Rate by Anomaly Flag
    ax2 = axes[1]
    df_plot = df.copy()
    df_plot["Pattern_Status"] = np.where(df_plot["Anomaly_Flag"] == 1, "Anomalous / High-Shift", "Normal Baseline")
    palette = {"Normal Baseline": "#1f77b4", "Anomalous / High-Shift": "#e377c2"}
    sns.boxplot(data=df_plot, x="Pattern_Status", y="Resistance_Rate", palette=palette, ax=ax2, width=0.45)
    ax2.set_title("Observed Resistance Rate by Anomaly Classification", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Prescription Pattern Status", fontsize=11)
    ax2.set_ylabel("Resistance Rate (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(out_dir, "isolation_forest_anomaly_analysis.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Isolation Forest analysis plot: {plot_path}")
    return plot_path

def plot_risk_score_distribution(df, out_dir):
    """Plots distribution of Composite Risk Scores and tier breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # 1. Risk Tier Counts
    ax1 = axes[0]
    tier_order = ["Low Risk", "Moderate Risk", "High Risk", "Critical Alert"]
    tier_counts = df["Risk_Tier"].value_counts().reindex(tier_order, fill_value=0)
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#7b1fa2"]
    
    bars = ax1.bar(tier_counts.index, tier_counts.values, color=colors, edgecolor="black", width=0.55)
    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., h + max(5, h*0.02), f"{int(h)} ({h/len(df)*100:.1f}%)",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
                 
    ax1.set_title("AMR Risk Tier Distribution Across Records", fontsize=13, fontweight="bold")
    ax1.set_ylabel("Number of Observations", fontsize=11)
    ax1.set_ylim(0, max(tier_counts.values) * 1.15)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="y")
    
    # 2. Risk Score vs Resistance Rate Scatter
    ax2 = axes[1]
    sns.scatterplot(
        data=df,
        x="Risk_Score",
        y="Predicted_Resistance_Rate" if "Predicted_Resistance_Rate" in df.columns else "Resistance_Rate",
        hue="Risk_Tier",
        palette=dict(zip(tier_order, colors)),
        hue_order=tier_order,
        alpha=0.65,
        s=40,
        ax=ax2
    )
    ax2.set_title("Composite Risk Score vs. Predicted Resistance Rate", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Composite AMR Risk Score (0 - 100)", fontsize=11)
    ax2.set_ylabel("Resistance Rate (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    
    plot_path = os.path.join(out_dir, "amr_risk_score_distribution.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved Risk Score distribution plot: {plot_path}")
    return plot_path

def generate_evaluation_report(metrics, fi_df, scored_df, out_dir):
    """Compiles a full textual and tabular summary report in markdown and csv."""
    report_md_path = os.path.join(out_dir, "model_evaluation_report.md")
    
    tier_summary = scored_df["Risk_Tier"].value_counts()
    anomaly_summary = scored_df["Anomaly_Flag"].value_counts()
    
    top_risk_locations = (
        scored_df.groupby("Location")
        .agg(
            Avg_Risk_Score=("Risk_Score", "mean"),
            Max_Risk_Score=("Risk_Score", "max"),
            Critical_Alerts=("Risk_Tier", lambda x: (x == "Critical Alert").sum()),
            High_Risks=("Risk_Tier", lambda x: (x == "High Risk").sum())
        )
        .sort_values("Avg_Risk_Score", ascending=False)
        .reset_index()
    )
    
    top_risk_locations.to_csv(os.path.join(out_dir, "location_risk_ranking.csv"), index=False)
    fi_df.to_csv(os.path.join(out_dir, "feature_importance_table.csv"), index=False)
    
    report_content = f"""# AMR Machine Learning System - Evaluation & Performance Report

## Executive Summary
This report summarizes the performance and findings of the combined **Isolation Forest Anomaly Detector** and **XGBoost Resistance Predictor**, along with the **Composite AMR Risk Scoring Engine**.

---

## 1. XGBoost Resistance Rate Prediction Performance
Evaluated on chronological out-of-time test partition:

| Metric | Test Value | Interpretation |
| :--- | :--- | :--- |
| **R² Score** | `{metrics['test_r2']:.4f}` | Model explains **{metrics['test_r2']*100:.2f}%** of the variance in resistance rates. |
| **Root Mean Squared Error (RMSE)** | `{metrics['test_rmse']:.2f}%` | Low average dispersion of prediction error. |
| **Mean Absolute Error (MAE)** | `{metrics['test_mae']:.2f}%` | Typical error is within **{metrics['test_mae']:.2f} percentage points** of resistance. |
| **Mean Absolute Percentage Error (MAPE)** | `{metrics['test_mape']:.2f}%` | Relative accuracy across drug classes. |

---

## 2. Isolation Forest Anomaly Detection Summary
- **Contamination Target**: 8%
- **Total Records Evaluated**: {len(scored_df)}
- **Identified Anomalous Prescribing Events**: {anomaly_summary.get(1, 0)} ({anomaly_summary.get(1, 0)/len(scored_df)*100:.2f}%)
- **Baseline Normal Events**: {anomaly_summary.get(0, 0)} ({anomaly_summary.get(0, 0)/len(scored_df)*100:.2f}%)

---

## 3. Top Feature Importances (XGBoost)
Top predictors driving resistance rate forecasting:

"""
    for idx, row in fi_df.head(10).iterrows():
        report_content += f"{idx+1}. **{row['Feature']}**: {row['Importance']:.4f}\n"

    report_content += f"""
---

## 4. AMR Composite Risk Tier Distribution
- **Critical Alert ($\ge 75$)**: {tier_summary.get('Critical Alert', 0)} ({tier_summary.get('Critical Alert', 0)/len(scored_df)*100:.2f}%)
- **High Risk (55–74.9)**: {tier_summary.get('High Risk', 0)} ({tier_summary.get('High Risk', 0)/len(scored_df)*100:.2f}%)
- **Moderate Risk (30–54.9)**: {tier_summary.get('Moderate Risk', 0)} ({tier_summary.get('Moderate Risk', 0)/len(scored_df)*100:.2f}%)
- **Low Risk ($< 30$)**: {tier_summary.get('Low Risk', 0)} ({tier_summary.get('Low Risk', 0)/len(scored_df)*100:.2f}%)

---

## 5. Top 5 Highest-Risk Surveillance Locations
"""
    for idx, row in top_risk_locations.head(5).iterrows():
        report_content += f"- **{row['Location']}**: Avg Risk Score = `{row['Avg_Risk_Score']:.2f}`, Critical Alerts = `{row['Critical_Alerts']}`\n"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Saved evaluation markdown report to: {report_md_path}")
    return report_md_path
