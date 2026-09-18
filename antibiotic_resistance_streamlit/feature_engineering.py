"""
Feature Engineering Module for Antimicrobial Resistance (AMR) ML System
Calculates WHO DDD metrics, class shifts, irrationality index, volatility,
spectrum exposure, and temporal lag features.
"""

import os
import pandas as pd
import numpy as np

# WHO Defined Daily Dose (DDD) in grams per unit standard
DDD_PER_UNIT = {
    "Amoxicillin": 1.5,              # g, oral
    "Amoxicillin-Clavulanate": 1.5,  # g, oral; refers to amoxicillin
    "Azithromycin": 0.3,             # g, oral
    "Cefixime": 0.4,                 # g, oral
    "Ceftriaxone": 2.0,              # g, parenteral
    "Ciprofloxacin": 1.0,            # g, oral
    "Doxycycline": 0.1,              # g, oral
    "Metronidazole": 2.0,            # g, oral
    "Cotrimoxazole": 1.92,           # g, standard oral DDD
    "Meropenem": 3.0,                # g, parenteral
}

# Spectrum numerical weight
SPECTRUM_WEIGHT = {
    "Narrow": 1,
    "Broad": 2,
    "Very Broad": 3
}

def load_raw_data(sales_path="pharmacy_sales.csv", resistance_path="resistance_data.csv"):
    """Loads and formats the raw CSV datasets."""
    sales = pd.read_csv(sales_path)
    resistance = pd.read_csv(resistance_path)
    
    sales["Month"] = pd.to_datetime(sales["Month"])
    resistance["Month"] = pd.to_datetime(resistance["Month"])
    
    sales["Quantity_Sold"] = pd.to_numeric(sales["Quantity_Sold"], errors="coerce").fillna(0)
    sales["Population"] = pd.to_numeric(sales["Population"], errors="coerce")
    sales["Is_Combination"] = pd.to_numeric(sales["Is_Combination"], errors="coerce").fillna(0)
    
    resistance["Resistance_Rate"] = pd.to_numeric(resistance["Resistance_Rate"], errors="coerce")
    
    return sales, resistance

def compute_location_month_metrics(sales):
    """
    Computes aggregate metrics at Location-Month level:
    - Total DDD / 1,000 inhabitants / day
    - Antibiotic Class Shift Index
    - Combination Irrationality Index
    - Volatility (Coefficient of Variation)
    """
    sales_df = sales.copy()
    sales_df["DDD_per_Unit"] = sales_df["Antibiotic"].map(DDD_PER_UNIT).fillna(1.0)
    sales_df["Total_DDD"] = sales_df["Quantity_Sold"] * sales_df["DDD_per_Unit"]
    
    # 1. Location-Month DDD/1000/day
    loc_month_ddd = (
        sales_df
        .groupby(["Location", "Month"])
        .agg(
            Total_Monthly_DDD=("Total_DDD", "sum"),
            Total_Monthly_Quantity=("Quantity_Sold", "sum"),
            Population=("Population", "first")
        )
        .reset_index()
    )
    loc_month_ddd["Days"] = loc_month_ddd["Month"].dt.days_in_month
    loc_month_ddd["DDD_per_1000_per_day"] = (
        loc_month_ddd["Total_Monthly_DDD"] * 1000 / (loc_month_ddd["Population"] * loc_month_ddd["Days"])
    )
    loc_month_ddd["Monthly_Quantity_per_1000"] = (
        loc_month_ddd["Total_Monthly_Quantity"] * 1000 / loc_month_ddd["Population"]
    )
    
    # 2. Antibiotic Class Shift
    class_monthly = (
        sales_df
        .groupby(["Location", "Month", "Class"])["Quantity_Sold"]
        .sum()
        .reset_index()
    )
    class_monthly["Total_Quantity"] = class_monthly.groupby(["Location", "Month"])["Quantity_Sold"].transform("sum")
    class_monthly["Class_Share"] = np.where(
        class_monthly["Total_Quantity"] > 0,
        class_monthly["Quantity_Sold"] / class_monthly["Total_Quantity"],
        0
    )
    
    class_monthly = class_monthly.sort_values(["Location", "Class", "Month"])
    class_monthly["Prev_Class_Share"] = class_monthly.groupby(["Location", "Class"])["Class_Share"].shift(1)
    class_monthly["Class_Share_Change"] = (class_monthly["Class_Share"] - class_monthly["Prev_Class_Share"]).fillna(0)
    
    class_shift = (
        class_monthly
        .groupby(["Location", "Month"])["Class_Share_Change"]
        .apply(lambda x: 0.5 * x.abs().sum())
        .reset_index(name="Antibiotic_Class_Shift")
    )
    class_shift["Antibiotic_Class_Shift_Percent"] = class_shift["Antibiotic_Class_Shift"] * 100
    
    # 3. Combination Irrationality Index
    combination_index = (
        sales_df
        .groupby(["Location", "Month"])
        .apply(
            lambda x: (
                x.loc[x["Is_Combination"] == 1, "Quantity_Sold"].sum() / x["Quantity_Sold"].sum()
                if x["Quantity_Sold"].sum() > 0 else 0
            )
        )
        .reset_index(name="Combination_Irrationality_Index")
    )
    combination_index["Combination_Irrationality_Index_Percent"] = combination_index["Combination_Irrationality_Index"] * 100
    
    # 4. Volatility (CV of monthly consumption per 1000)
    volatility = (
        loc_month_ddd
        .groupby("Location")
        .agg(
            Mean_Quantity_per_1000=("Monthly_Quantity_per_1000", "mean"),
            SD_Quantity_per_1000=("Monthly_Quantity_per_1000", "std")
        )
        .reset_index()
    )
    volatility["Volatility_CV"] = np.where(
        volatility["Mean_Quantity_per_1000"] > 0,
        volatility["SD_Quantity_per_1000"].fillna(0) / volatility["Mean_Quantity_per_1000"],
        0
    )
    volatility["Volatility_Percent"] = volatility["Volatility_CV"] * 100
    
    # Merge Location-Month macro features
    loc_month_features = loc_month_ddd.merge(class_shift, on=["Location", "Month"], how="left")
    loc_month_features = loc_month_features.merge(combination_index, on=["Location", "Month"], how="left")
    loc_month_features = loc_month_features.merge(
        volatility[["Location", "Volatility_CV", "Volatility_Percent"]], on="Location", how="left"
    )
    
    return loc_month_features

def build_complete_feature_matrix(sales_path="pharmacy_sales.csv", resistance_path="resistance_data.csv"):
    """
    Builds the combined, granular dataset merged with resistance rates
    and enriched with temporal lags and domain metrics.
    """
    sales, resistance = load_raw_data(sales_path, resistance_path)
    loc_month_features = compute_location_month_metrics(sales)
    
    # Antibiotic level metrics
    df = sales.copy()
    df["DDD_per_Unit"] = df["Antibiotic"].map(DDD_PER_UNIT).fillna(1.0)
    df["Total_DDD"] = df["Quantity_Sold"] * df["DDD_per_Unit"]
    df["Quantity_per_1000"] = df["Quantity_Sold"] * 1000 / df["Population"]
    df["Spectrum_Weight"] = df["Spectrum"].map(SPECTRUM_WEIGHT).fillna(2)
    
    # Calculate Antibiotic share within monthly location usage
    loc_totals = df.groupby(["Location", "Month"])["Quantity_Sold"].sum().reset_index(name="Loc_Total_Quantity")
    df = df.merge(loc_totals, on=["Location", "Month"], how="left")
    df["Antibiotic_Share"] = np.where(
        df["Loc_Total_Quantity"] > 0,
        df["Quantity_Sold"] / df["Loc_Total_Quantity"],
        0
    )
    
    # Merge with resistance data
    df = df.merge(resistance, on=["Location", "Month", "Antibiotic"], how="inner")
    
    # Merge Location-Month macro features
    df = df.merge(
        loc_month_features[[
            "Location", "Month", "Total_Monthly_DDD", "Total_Monthly_Quantity",
            "DDD_per_1000_per_day", "Antibiotic_Class_Shift", "Antibiotic_Class_Shift_Percent",
            "Combination_Irrationality_Index", "Combination_Irrationality_Index_Percent",
            "Volatility_CV", "Volatility_Percent"
        ]],
        on=["Location", "Month"],
        how="left"
    )
    
    # Sort for time-series / temporal lag creation
    df = df.sort_values(["Location", "Antibiotic", "Month"]).reset_index(drop=True)
    
    # Temporal Lag Features (Antibiotic specific within location)
    df["Lag1_Quantity_Sold"] = df.groupby(["Location", "Antibiotic"])["Quantity_Sold"].shift(1)
    df["Lag2_Quantity_Sold"] = df.groupby(["Location", "Antibiotic"])["Quantity_Sold"].shift(2)
    df["Lag1_Resistance_Rate"] = df.groupby(["Location", "Antibiotic"])["Resistance_Rate"].shift(1)
    df["Lag2_Resistance_Rate"] = df.groupby(["Location", "Antibiotic"])["Resistance_Rate"].shift(2)
    
    # Rolling averages (3-month window)
    df["Rolling_3M_Quantity"] = (
        df.groupby(["Location", "Antibiotic"])["Quantity_Sold"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    df["Rolling_3M_Resistance"] = (
        df.groupby(["Location", "Antibiotic"])["Resistance_Rate"]
        .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )
    
    # Fill backward/forward lags where appropriate
    df["Lag1_Quantity_Sold"] = df["Lag1_Quantity_Sold"].fillna(df["Quantity_Sold"])
    df["Lag2_Quantity_Sold"] = df["Lag2_Quantity_Sold"].fillna(df["Lag1_Quantity_Sold"])
    df["Lag1_Resistance_Rate"] = df["Lag1_Resistance_Rate"].fillna(df["Resistance_Rate"])
    df["Lag2_Resistance_Rate"] = df["Lag2_Resistance_Rate"].fillna(df["Lag1_Resistance_Rate"])
    
    # Month numerical features
    df["Month_Number"] = df["Month"].dt.month
    df["Year"] = df["Month"].dt.year
    df["Quarter"] = df["Month"].dt.quarter
    
    # Clean infs and NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df, loc_month_features

if __name__ == "__main__":
    print("Executing feature engineering pipeline...")
    data_dir = os.path.dirname(os.path.abspath(__file__))
    sales_file = os.path.join(data_dir, "pharmacy_sales.csv")
    resistance_file = os.path.join(data_dir, "resistance_data.csv")
    
    processed_df, loc_month_df = build_complete_feature_matrix(sales_file, resistance_file)
    
    out_dir = os.path.join(data_dir, "data")
    os.makedirs(out_dir, exist_ok=True)
    
    processed_df.to_csv(os.path.join(out_dir, "processed_features.csv"), index=False)
    loc_month_df.to_csv(os.path.join(out_dir, "location_month_features.csv"), index=False)
    
    print(f"Feature engineering successful! Processed shape: {processed_df.shape}")
    print(f"Columns: {list(processed_df.columns)}")
