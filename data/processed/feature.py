import pandas as pd
import numpy as np


# ============================================================
# LOAD DATA
# ============================================================

sales = pd.read_csv("pharmacy_sales.csv")

sales["Month"] = pd.to_datetime(sales["Month"])

sales["Quantity_Sold"] = pd.to_numeric(
    sales["Quantity_Sold"],
    errors="coerce"
)

sales["Population"] = pd.to_numeric(
    sales["Population"],
    errors="coerce"
)

sales["Is_Combination"] = pd.to_numeric(
    sales["Is_Combination"],
    errors="coerce"
).fillna(0)


# ============================================================
# 1. DDD / 1,000 INHABITANTS / DAY
# ============================================================


DDD_PER_UNIT = {
    "Amoxicillin": 1.5,              # g, oral
    "Amoxicillin-Clavulanate": 1.5,  # g, oral; refers to amoxicillin
    "Azithromycin": 0.3,             # g, oral
    "Cefixime": 0.4,                 # g, oral
    "Ceftriaxone": 2.0,              # g, parenteral
    "Ciprofloxacin": 1.0,            # g, oral
    "Doxycycline": 0.1,              # g, oral
    "Metronidazole": 2.0,            # g, oral
    "Cotrimoxazole": None,            # no standard DDD assigned
    "Meropenem": 3.0,                # g, parenteral
}


if DDD_PER_UNIT:

    sales["DDD_per_Unit"] = sales["Antibiotic"].map(
        DDD_PER_UNIT
    )

    sales["Total_DDD"] = (
        sales["Quantity_Sold"]
        * sales["DDD_per_Unit"]
    )

    ddd_1000 = (
        sales
        .groupby(["Location", "Month"])
        .agg(
            Total_DDD=("Total_DDD", "sum"),
            Population=("Population", "first")
        )
        .reset_index()
    )

    ddd_1000["Days"] = (
        ddd_1000["Month"].dt.days_in_month
    )

    ddd_1000["DDD_per_1000_per_day"] = (
        ddd_1000["Total_DDD"]
        * 1000
        / (
            ddd_1000["Population"]
            * ddd_1000["Days"]
        )
    )

else:

    print(
        "DDD/1000/day cannot be calculated yet."
    )
    ddd_1000 = None


# ============================================================
# 2. ANTIBIOTIC CLASS SHIFT
# ============================================================

# First calculate the quantity used in each antibiotic class
# every month.

class_monthly = (
    sales
    .groupby(
        [
            "Location",
            "Month",
            "Class"
        ]
    )["Quantity_Sold"]
    .sum()
    .reset_index()
)


# Total quantity per location/month

class_monthly["Total_Quantity"] = (
    class_monthly
    .groupby(
        ["Location", "Month"]
    )["Quantity_Sold"]
    .transform("sum")
)


# Proportion of each class

class_monthly["Class_Share"] = (
    class_monthly["Quantity_Sold"]
    / class_monthly["Total_Quantity"]
)


# Compare current month with previous month

class_monthly = class_monthly.sort_values(
    [
        "Location",
        "Class",
        "Month"
    ]
)

class_monthly["Previous_Class_Share"] = (
    class_monthly
    .groupby(
        ["Location", "Class"]
    )["Class_Share"]
    .shift(1)
)


# Difference in class share

class_monthly["Class_Share_Change"] = (
    class_monthly["Class_Share"]
    - class_monthly["Previous_Class_Share"]
)


# ------------------------------------------------------------
# Class Shift Index
# ------------------------------------------------------------
#
# Class Shift =
# 0.5 × sum of absolute changes in class proportions
#
# 0   = no change
# 1   = complete redistribution
#
# Expressed as percentage below.

class_shift = (
    class_monthly
    .groupby(
        ["Location", "Month"]
    )["Class_Share_Change"]
    .apply(
        lambda x: 0.5 * x.dropna().abs().sum()
    )
    .reset_index(
        name="Antibiotic_Class_Shift"
    )
)

class_shift["Antibiotic_Class_Shift_Percent"] = (
    class_shift["Antibiotic_Class_Shift"]
    * 100
)


# ============================================================
# 3. COMBINATION IRRATIONALITY INDEX
# ============================================================

# Your dataset contains Is_Combination.
#
# Therefore we can calculate:
#
# Combination use / Total antibiotic use × 100
#
# This is a PROXY for combination irrationality.
#
# It should NOT be called definitively "irrational"
# unless you have a clinical/pharmacological rule identifying
# which combinations are irrational.

combination_index = (
    sales
    .groupby(
        ["Location", "Month"]
    )
    .apply(
        lambda x: (
            x.loc[
                x["Is_Combination"] == 1,
                "Quantity_Sold"
            ].sum()
            /
            x["Quantity_Sold"].sum()
        )
        if x["Quantity_Sold"].sum() > 0
        else np.nan
    )
    .reset_index(
        name="Combination_Irrationality_Index"
    )
)


combination_index[
    "Combination_Irrationality_Index_Percent"
] = (
    combination_index[
        "Combination_Irrationality_Index"
    ]
    * 100
)


# ============================================================
# 4. VOLATILITY
# ============================================================

# We calculate volatility of antibiotic consumption
# using the coefficient of variation:
#
# Volatility = SD / Mean
#
# calculated across the available monthly observations.

monthly_quantity = (
    sales
    .groupby(
        ["Location", "Month"]
    )
    .agg(
        Total_Quantity=("Quantity_Sold", "sum"),
        Population=("Population", "first")
    )
    .reset_index()
)


monthly_quantity["Quantity_per_1000"] = (
    monthly_quantity["Total_Quantity"]
    * 1000
    / monthly_quantity["Population"]
)


volatility = (
    monthly_quantity
    .groupby("Location")
    .agg(
        Mean_Quantity_per_1000=(
            "Quantity_per_1000",
            "mean"
        ),

        SD_Quantity_per_1000=(
            "Quantity_per_1000",
            "std"
        )
    )
    .reset_index()
)


volatility["Volatility_CV"] = np.where(
    volatility["Mean_Quantity_per_1000"] > 0,

    volatility["SD_Quantity_per_1000"]
    / volatility["Mean_Quantity_per_1000"],

    np.nan
)


volatility["Volatility_Percent"] = (
    volatility["Volatility_CV"]
    * 100
)


# ============================================================
# 5. CREATE FINAL FEATURE DATASET
# ============================================================

features = monthly_quantity[
    [
        "Location",
        "Month",
        "Total_Quantity",
        "Population",
        "Quantity_per_1000"
    ]
].copy()


# Add class shift

features = features.merge(
    class_shift[
        [
            "Location",
            "Month",
            "Antibiotic_Class_Shift",
            "Antibiotic_Class_Shift_Percent"
        ]
    ],
    on=[
        "Location",
        "Month"
    ],
    how="left"
)


# Add combination index

features = features.merge(
    combination_index[
        [
            "Location",
            "Month",
            "Combination_Irrationality_Index",
            "Combination_Irrationality_Index_Percent"
        ]
    ],
    on=[
        "Location",
        "Month"
    ],
    how="left"
)


# Add location-level volatility

features = features.merge(
    volatility[
        [
            "Location",
            "Volatility_CV",
            "Volatility_Percent"
        ]
    ],
    on="Location",
    how="left"
)


# Add DDD if available

if ddd_1000 is not None:

    features = features.merge(
        ddd_1000[
            [
                "Location",
                "Month",
                "DDD_per_1000_per_day"
            ]
        ],
        on=[
            "Location",
            "Month"
        ],
        how="left"
    )


# ============================================================
# 6. CLEAN FINAL FEATURE DATASET
# ============================================================

features = features.replace(
    [np.inf, -np.inf],
    np.nan
)

features = features.sort_values(
    [
        "Location",
        "Month"
    ]
).reset_index(drop=True)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n==========================================")
print("FEATURE ENGINEERING COMPLETED")
print("==========================================")

print("\nFeatures:")
print(features.columns.tolist())

print("\nFinal feature dataset:")
print(features.head(20))