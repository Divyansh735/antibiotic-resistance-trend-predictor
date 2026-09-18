
import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

# Make the bundled teammate modules importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(
    page_title="AMR Trend Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
    .main {background: #f7f9fc;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.15rem;}
    .hero p {margin: .35rem 0 0; opacity: .86;}
    .metric-card {
        background: white; border: 1px solid #e5e7eb; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 2px 10px rgba(15,23,42,.04);
    }
    .small-muted {color:#64748b; font-size:.86rem;}
    .alert-box {
        border-left: 5px solid #ef4444; background:#fff7f7; padding:.85rem 1rem;
        border-radius: 8px; margin:.4rem 0;
    }
    .info-box {
        border-left: 5px solid #2563eb; background:#eff6ff; padding:.85rem 1rem;
        border-radius: 8px; margin:.5rem 0;
    }
    [data-testid="stSidebar"] {background: #ffffff;}
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_FILE = os.path.join(DATA_DIR, "amr_complete_predictions_and_risks.csv")

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    if "Month" in df.columns:
        df["Month"] = df["Month"].astype(str)
        try:
            df["_date"] = pd.to_datetime(df["Month"])
        except Exception:
            df["_date"] = pd.NaT
    return df

@st.cache_resource
def train_live_xgb():
    """Train the same XGBoost pipeline supplied by the project team.
    This is only used by the optional User Prediction section.
    """
    try:
        from xgboost_model import AMRResistanceXGBoost
        df = load_data(DEFAULT_FILE)
        model = AMRResistanceXGBoost()
        metrics, _, _ = model.train_and_evaluate_temporal_split(df)
        return model, metrics, None
    except Exception as e:
        return None, None, str(e)

def risk_color(level):
    return {
        "Low Risk": "🟢",
        "Moderate Risk": "🟡",
        "High Risk": "🟠",
        "Critical Alert": "🔴"
    }.get(level, "⚪")

def fmt(v, digits=2):
    try:
        return f"{float(v):,.{digits}f}"
    except Exception:
        return "—"

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>🧬 Antibiotic Resistance Trend Predictor</h1>
    <p>AMR surveillance dashboard • Pharmacy consumption • Risk analytics • XGBoost prediction</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    uploaded = st.file_uploader(
        "Upload project output CSV",
        type=["csv"],
        help="Recommended: amr_complete_predictions_and_risks.csv"
    )
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if "Location" not in df.columns or "Risk_Score" not in df.columns:
                st.error("This CSV needs at least Location and Risk_Score columns.")
                df = load_data(DEFAULT_FILE)
            else:
                if "Month" in df.columns:
                    df["Month"] = df["Month"].astype(str)
                    df["_date"] = pd.to_datetime(df["Month"], errors="coerce")
                st.success("Uploaded dataset loaded.")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = load_data(DEFAULT_FILE)
    else:
        df = load_data(DEFAULT_FILE)

    st.divider()
    locations = ["All"] + sorted(df["Location"].dropna().unique().tolist())
    selected_location = st.selectbox("📍 District / Location", locations)

    antibiotics = ["All"] + sorted(df["Antibiotic"].dropna().unique().tolist())
    selected_antibiotic = st.selectbox("💊 Antibiotic", antibiotics)

    months = ["All"] + sorted(df["Month"].dropna().unique().tolist())
    selected_month = st.selectbox("📅 Month", months, index=len(months)-1 if len(months)>1 else 0)

    st.divider()
    st.caption("Project prototype for academic/exhibition use.")
    st.caption("Risk scores are model outputs, not clinical decisions.")

# ---------- Filters ----------
filtered = df.copy()
if selected_location != "All":
    filtered = filtered[filtered["Location"] == selected_location]
if selected_antibiotic != "All":
    filtered = filtered[filtered["Antibiotic"] == selected_antibiotic]
if selected_month != "All":
    filtered = filtered[filtered["Month"] == selected_month]

if filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# ---------- Overview ----------
st.subheader("📊 Project Overview")
c1,c2,c3,c4 = st.columns(4)
with c1:
    st.metric("Records", f"{len(filtered):,}")
with c2:
    st.metric("Locations", f"{filtered['Location'].nunique():,}")
with c3:
    st.metric("Avg Risk Score", fmt(filtered["Risk_Score"].mean()))
with c4:
    st.metric("Avg Predicted Resistance", f"{fmt(filtered['Predicted_Resistance_Rate'].mean())}%")

st.markdown('<div class="info-box"><b>How the pipeline works:</b> pharmacy sales → feature engineering → Isolation Forest anomaly detection → composite AMR risk score → XGBoost resistance prediction → dashboard alerts.</div>', unsafe_allow_html=True)

# ---------- Current risk snapshot ----------
st.subheader("🎯 Current Risk Snapshot")
latest = filtered.copy()
if selected_month == "All" and "_date" in latest.columns and latest["_date"].notna().any():
    latest_month = latest["_date"].max()
    latest = latest[latest["_date"] == latest_month]

m1,m2,m3,m4 = st.columns(4)
avg_score = latest["Risk_Score"].mean()
max_score = latest["Risk_Score"].max()
avg_res = latest["Predicted_Resistance_Rate"].mean()
top_tier = latest["Risk_Tier"].mode().iloc[0] if "Risk_Tier" in latest and not latest["Risk_Tier"].empty else "—"

m1.metric("Average Risk", fmt(avg_score))
m2.metric("Maximum Risk", fmt(max_score))
m3.metric("Predicted Resistance", f"{fmt(avg_res)}%")
m4.metric("Dominant Risk Tier", f"{risk_color(top_tier)} {top_tier}")

# ---------- Risk distribution ----------
left,right = st.columns([1.05, 1])
with left:
    st.markdown("#### Risk Tier Distribution")
    if "Risk_Tier" in filtered:
        tier_order = ["Low Risk","Moderate Risk","High Risk","Critical Alert"]
        counts = filtered["Risk_Tier"].value_counts().reindex(tier_order, fill_value=0).reset_index()
        counts.columns = ["Risk Tier","Count"]
        fig = px.bar(counts, x="Risk Tier", y="Count", text="Count")
        fig.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### District Risk Ranking")
    if "Location" in df.columns:
        ranking = (
            df.groupby("Location", as_index=False)
            .agg(Avg_Risk_Score=("Risk_Score","mean"),
                 Max_Risk_Score=("Risk_Score","max"),
                 Avg_Predicted_Resistance=("Predicted_Resistance_Rate","mean"))
            .sort_values("Avg_Risk_Score", ascending=False)
            .head(10)
        )
        fig2 = px.bar(ranking.sort_values("Avg_Risk_Score"), x="Avg_Risk_Score", y="Location",
                      orientation="h", text="Avg_Risk_Score")
        fig2.update_layout(height=330, margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

# ---------- Trend charts ----------
st.subheader("📈 AMR Indicator Trends")
if selected_month == "All":
    trend = filtered.groupby("Month", as_index=False).agg(
        DDD_per_1000_per_day=("DDD_per_1000_per_day","mean"),
        Broad_Spectrum_Ratio=("Spectrum_Weight", lambda x: np.nan),
        Combination_Index=("Combination_Irrationality_Index_Percent","mean"),
        Volatility=("Volatility_Percent","mean"),
        Risk_Score=("Risk_Score","mean"),
        Predicted_Resistance=("Predicted_Resistance_Rate","mean")
    )
    # Broad spectrum ratio: share of records marked Broad/Very Broad, weighted by row count.
    temp = filtered.copy()
    temp["Broad_Flag"] = temp["Spectrum"].isin(["Broad","Very Broad"]).astype(int)
    broad = temp.groupby("Month")["Broad_Flag"].mean().mul(100).rename("Broad_Spectrum_Ratio")
    trend = trend.drop(columns=["Broad_Spectrum_Ratio"]).merge(broad, on="Month", how="left")
    trend = trend.sort_values("Month")
else:
    trend = filtered.copy()
    trend["Broad_Spectrum_Ratio"] = trend["Spectrum"].isin(["Broad","Very Broad"]).astype(int)*100
    trend = trend.groupby("Month", as_index=False).agg(
        DDD_per_1000_per_day=("DDD_per_1000_per_day","mean"),
        Broad_Spectrum_Ratio=("Broad_Spectrum_Ratio","mean"),
        Combination_Index=("Combination_Irrationality_Index_Percent","mean"),
        Volatility=("Volatility_Percent","mean"),
        Risk_Score=("Risk_Score","mean"),
        Predicted_Resistance=("Predicted_Resistance_Rate","mean")
    )

tabs = st.tabs(["DDD / 1000 / day", "Broad-spectrum ratio", "Combination index", "Volatility", "Resistance"])
with tabs[0]:
    fig = px.line(trend, x="Month", y="DDD_per_1000_per_day", markers=True)
    fig.update_layout(height=320, yaxis_title="DDD / 1000 / day")
    st.plotly_chart(fig, use_container_width=True)
with tabs[1]:
    fig = px.line(trend, x="Month", y="Broad_Spectrum_Ratio", markers=True)
    fig.update_layout(height=320, yaxis_title="Broad / Very Broad share (%)")
    st.plotly_chart(fig, use_container_width=True)
with tabs[2]:
    fig = px.line(trend, x="Month", y="Combination_Index", markers=True)
    fig.update_layout(height=320, yaxis_title="Combination index (%)")
    st.plotly_chart(fig, use_container_width=True)
with tabs[3]:
    fig = px.line(trend, x="Month", y="Volatility", markers=True)
    fig.update_layout(height=320, yaxis_title="Volatility (%)")
    st.plotly_chart(fig, use_container_width=True)
with tabs[4]:
    fig = px.line(trend, x="Month", y="Predicted_Resistance", markers=True)
    fig.update_layout(height=320, yaxis_title="Predicted resistance (%)")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Alerts ----------
st.subheader("🚨 Early-warning Alerts")
alerts = filtered[filtered["Risk_Tier"].isin(["High Risk","Critical Alert"])].copy()
if alerts.empty:
    moderate = filtered[filtered["Risk_Tier"]=="Moderate Risk"].sort_values("Risk_Score", ascending=False).head(5)
    if moderate.empty:
        st.success("No high/critical risk records found in the current filter.")
    else:
        st.info("No High/Critical records in this filter. Showing the highest Moderate Risk records for monitoring.")
        for _, r in moderate.iterrows():
            st.markdown(f'<div class="alert-box"><b>{r["Location"]} • {r["Antibiotic"]} • {r["Month"]}</b><br>Risk score: <b>{fmt(r["Risk_Score"])}</b> • Predicted resistance: <b>{fmt(r["Predicted_Resistance_Rate"])}%</b></div>', unsafe_allow_html=True)
else:
    for _, r in alerts.sort_values("Risk_Score", ascending=False).head(8).iterrows():
        action = r.get("Recommended_Action","Review the pattern.")
        st.markdown(f'<div class="alert-box"><b>{risk_color(r["Risk_Tier"])} {r["Location"]} • {r["Antibiotic"]} • {r["Month"]}</b><br>Risk score: <b>{fmt(r["Risk_Score"])}</b> • Predicted resistance: <b>{fmt(r["Predicted_Resistance_Rate"])}%</b><br><span class="small-muted">{action}</span></div>', unsafe_allow_html=True)

# ---------- Major risk factors ----------
st.subheader("🔎 Major Risk Factors")
factor_cols = [
    ("DDD / 1000 / day","DDD_per_1000_per_day"),
    ("Broad-spectrum share","Spectrum"),
    ("Combination index","Combination_Irrationality_Index_Percent"),
    ("Volatility","Volatility_Percent"),
    ("Anomaly score","Anomaly_Risk_Score"),
    ("Class shift","Antibiotic_Class_Shift_Percent")
]
factor_values = []
for label,col in factor_cols:
    if col == "Spectrum":
        val = filtered[col].isin(["Broad","Very Broad"]).mean()*100
    else:
        val = filtered[col].mean()
    factor_values.append({"Factor":label,"Average":val})
factor_df = pd.DataFrame(factor_values)
fig = px.bar(factor_df, x="Average", y="Factor", orientation="h", text="Average")
fig.update_layout(height=310, margin=dict(l=10,r=10,t=10,b=10))
st.plotly_chart(fig, use_container_width=True)

# ---------- Prediction section ----------
st.subheader("🤖 User Input / XGBoost Prediction")
st.caption("This section trains the supplied project XGBoost pipeline on the bundled project data and predicts resistance for an edited historical profile. It is intended as a project demonstration, not a clinical forecasting tool.")

try:
    base_rows = df.copy()
    loc_choices = sorted(base_rows["Location"].unique())
    drug_choices = sorted(base_rows["Antibiotic"].unique())
    p1,p2,p3 = st.columns(3)
    with p1:
        inp_loc = st.selectbox("Location", loc_choices, key="pred_loc")
    with p2:
        inp_drug = st.selectbox("Antibiotic", drug_choices, key="pred_drug")
    subset = base_rows[(base_rows["Location"]==inp_loc) & (base_rows["Antibiotic"]==inp_drug)].copy()
    if subset.empty:
        st.warning("No profile found.")
    else:
        subset["_date"] = pd.to_datetime(subset["Month"], errors="coerce")
        base = subset.sort_values("_date").iloc[-1].copy()
        with p3:
            st.text_input("Base month", value=str(base["Month"]), disabled=True)

        q_col, comb_col = st.columns(2)
        with q_col:
            new_qty = st.number_input("Quantity sold", min_value=0.0, value=float(base["Quantity_Sold"]), step=100.0)
        with comb_col:
            st.number_input("Combination flag (0/1)", min_value=0, max_value=1, value=int(base["Is_Combination"]), step=1, disabled=True)

        if st.button("🔮 Predict Resistance", type="primary"):
            with st.spinner("Running the project XGBoost pipeline..."):
                model, metrics, error = train_live_xgb()
            if error:
                st.error(f"XGBoost could not be loaded/trained: {error}")
            else:
                row = base.to_frame().T.copy()
                row["Quantity_Sold"] = new_qty
                row["Quantity_per_1000"] = row["Quantity_Sold"] * 1000 / row["Population"]
                # Recompute the two directly dependent consumption features for the demo row.
                row["Total_DDD"] = row["Quantity_Sold"] * row["Total_DDD"] / max(float(base["Quantity_Sold"]), 1.0)
                row["Rolling_3M_Quantity"] = (float(base["Rolling_3M_Quantity"])*2 + new_qty)/3
                pred = float(model.predict(row)["Predicted_Resistance_Rate"].iloc[0])
                st.success(f"Predicted resistance rate: **{pred:.2f}%**")
                st.caption(f"Model test R² from the supplied pipeline: {metrics['test_r2']:.4f} | RMSE: {metrics['test_rmse']:.2f}%")
except Exception as e:
    st.info(f"Live prediction section is unavailable until the project ML dependencies are installed. Details: {e}")

# ---------- Data table ----------
with st.expander("📋 View filtered records"):
    display_cols = [c for c in [
        "Location","Month","Antibiotic","Spectrum","Quantity_Sold",
        "DDD_per_1000_per_day","Combination_Irrationality_Index_Percent",
        "Volatility_Percent","Predicted_Resistance_Rate","Risk_Score",
        "Risk_Tier","Recommended_Action"
    ] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, hide_index=True)

# ---------- Footer ----------
st.divider()
st.caption("Antibiotic Resistance Trend Predictor • VIT Bhopal • Academic Project Prototype")
st.caption("Interpretation: Higher risk scores indicate patterns flagged by the project's composite scoring framework. This dashboard does not replace microbiological surveillance or clinical judgement.")
