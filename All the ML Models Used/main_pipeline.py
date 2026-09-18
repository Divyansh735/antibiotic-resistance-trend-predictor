"""
AMR Machine Learning Pipeline: Feature Engineering, Isolation Forest,
Gradient Boosted Regressor (XGBoost Architecture), Risk Scoring, and Evaluation.
"""

import os
import sys
import csv
import math
import json
import datetime
import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

DDD_PER_UNIT = {
    "Amoxicillin": 1.5,
    "Amoxicillin-Clavulanate": 1.5,
    "Azithromycin": 0.3,
    "Cefixime": 0.4,
    "Ceftriaxone": 2.0,
    "Ciprofloxacin": 1.0,
    "Doxycycline": 0.1,
    "Metronidazole": 2.0,
    "Cotrimoxazole": 1.92,
    "Meropenem": 3.0
}

SPECTRUM_MAP = {
    "Narrow": 1,
    "Broad": 2,
    "Very Broad": 3
}

SPECTRUM_FACTOR = {
    "Narrow": 30.0,
    "Broad": 70.0,
    "Very Broad": 100.0
}

def get_days_in_month(year, month):
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    elif month in (4, 6, 9, 11):
        return 30
    elif month == 2:
        return 29 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 28
    return 30

# ==============================================================================
# 1. FEATURE ENGINEERING
# ==============================================================================

def load_and_engineer_features(sales_path, resistance_path):
    print("  -> Loading raw CSV data...")
    with open(sales_path, "r", encoding="utf-8") as f:
        sales_rows = list(csv.DictReader(f))
    with open(resistance_path, "r", encoding="utf-8") as f:
        res_rows = list(csv.DictReader(f))
        
    res_dict = {}
    for r in res_rows:
        key = (r["Location"].strip(), r["Month"].strip(), r["Antibiotic"].strip())
        try:
            res_dict[key] = float(r["Resistance_Rate"])
        except ValueError:
            res_dict[key] = 0.0

    # Parse and compute unit DDDs
    parsed_sales = []
    for r in sales_rows:
        loc = r["Location"].strip()
        month_str = r["Month"].strip()
        parts = month_str.split("-")
        year = int(parts[0])
        month_num = int(parts[1])
        antibiotic = r["Antibiotic"].strip()
        drug_class = r.get("Class", "").strip()
        spectrum = r.get("Spectrum", "Broad").strip()
        
        try:
            pop = float(r.get("Population", 1000000))
        except ValueError:
            pop = 1000000.0
            
        try:
            is_comb = float(r.get("Is_Combination", 0))
        except ValueError:
            is_comb = 0.0
            
        try:
            qty = float(r.get("Quantity_Sold", 0))
        except ValueError:
            qty = 0.0
            
        try:
            lat = float(r.get("Latitude", 0))
            lon = float(r.get("Longitude", 0))
        except ValueError:
            lat, lon = 0.0, 0.0
            
        ddd_unit = DDD_PER_UNIT.get(antibiotic, 1.0)
        total_ddd = qty * ddd_unit
        spectrum_weight = SPECTRUM_MAP.get(spectrum, 2)
        
        res_rate = res_dict.get((loc, month_str, antibiotic), 0.0)
        
        parsed_sales.append({
            "Location": loc,
            "Month": month_str,
            "Year": year,
            "Month_Number": month_num,
            "Quarter": (month_num - 1) // 3 + 1,
            "Days": get_days_in_month(year, month_num),
            "Population": pop,
            "Antibiotic": antibiotic,
            "Class": drug_class,
            "Spectrum": spectrum,
            "Spectrum_Weight": spectrum_weight,
            "Is_Combination": is_comb,
            "Quantity_Sold": qty,
            "Total_DDD": total_ddd,
            "Latitude": lat,
            "Longitude": lon,
            "Resistance_Rate": res_rate
        })

    # Group by Location-Month
    loc_month_map = {}
    for r in parsed_sales:
        lm_key = (r["Location"], r["Month"])
        if lm_key not in loc_month_map:
            loc_month_map[lm_key] = {
                "Total_Monthly_Quantity": 0.0,
                "Total_Monthly_DDD": 0.0,
                "Population": r["Population"],
                "Days": r["Days"],
                "Combination_Quantity": 0.0,
                "Class_Quantities": {}
            }
        loc_month_map[lm_key]["Total_Monthly_Quantity"] += r["Quantity_Sold"]
        loc_month_map[lm_key]["Total_Monthly_DDD"] += r["Total_DDD"]
        if r["Is_Combination"] == 1:
            loc_month_map[lm_key]["Combination_Quantity"] += r["Quantity_Sold"]
            
        cls = r["Class"]
        loc_month_map[lm_key]["Class_Quantities"][cls] = loc_month_map[lm_key]["Class_Quantities"].get(cls, 0.0) + r["Quantity_Sold"]

    # Compute DDD/1000/day and Combination Irrationality Index
    for lm_key, metrics in loc_month_map.items():
        pop = metrics["Population"]
        days = metrics["Days"]
        tot_qty = metrics["Total_Monthly_Quantity"]
        tot_ddd = metrics["Total_Monthly_DDD"]
        
        metrics["DDD_per_1000_per_day"] = (tot_ddd * 1000.0) / (pop * days) if (pop * days) > 0 else 0.0
        metrics["Monthly_Quantity_per_1000"] = (tot_qty * 1000.0) / pop if pop > 0 else 0.0
        metrics["Combination_Irrationality_Index_Percent"] = (metrics["Combination_Quantity"] / tot_qty * 100.0) if tot_qty > 0 else 0.0

    # Compute Class Shift
    loc_months = {}
    for (loc, m), metrics in loc_month_map.items():
        if loc not in loc_months:
            loc_months[loc] = []
        loc_months[loc].append((m, metrics))
        
    for loc, m_list in loc_months.items():
        m_list.sort(key=lambda x: x[0])
        prev_shares = {}
        for m_str, m_data in m_list:
            tot = m_data["Total_Monthly_Quantity"]
            curr_shares = {c: (q / tot if tot > 0 else 0.0) for c, q in m_data["Class_Quantities"].items()}
            
            if prev_shares:
                all_classes = set(curr_shares.keys()).union(set(prev_shares.keys()))
                shift_val = 0.5 * sum(abs(curr_shares.get(c, 0.0) - prev_shares.get(c, 0.0)) for c in all_classes)
                class_shift_pct = shift_val * 100.0
            else:
                class_shift_pct = 0.0
                
            m_data["Antibiotic_Class_Shift_Percent"] = class_shift_pct
            prev_shares = curr_shares

    # Compute Location-level Volatility
    loc_volatility = {}
    for loc, m_list in loc_months.items():
        q_per_1000 = [m_data["Monthly_Quantity_per_1000"] for _, m_data in m_list]
        mean_q = float(np.mean(q_per_1000))
        std_q = float(np.std(q_per_1000))
        cv = (std_q / mean_q * 100.0) if mean_q > 0 else 0.0
        loc_volatility[loc] = cv

    # Enforce temporal sorting for lags
    parsed_sales.sort(key=lambda x: (x["Location"], x["Antibiotic"], x["Month"]))
    
    # Calculate antibiotic-level lags & temporal rolling metrics
    records = []
    drug_loc_history = {}
    
    for r in parsed_sales:
        loc = r["Location"]
        month_str = r["Month"]
        antibiotic = r["Antibiotic"]
        lm_data = loc_month_map[(loc, month_str)]
        
        tot_monthly_qty = lm_data["Total_Monthly_Quantity"]
        antibiotic_share = (r["Quantity_Sold"] / tot_monthly_qty) if tot_monthly_qty > 0 else 0.0
        qty_per_1000 = (r["Quantity_Sold"] * 1000.0) / r["Population"] if r["Population"] > 0 else 0.0
        
        key = (loc, antibiotic)
        if key not in drug_loc_history:
            drug_loc_history[key] = {
                "quantities": [],
                "resistances": []
            }
            
        hist = drug_loc_history[key]
        lag1_qty = hist["quantities"][-1] if len(hist["quantities"]) >= 1 else r["Quantity_Sold"]
        lag2_qty = hist["quantities"][-2] if len(hist["quantities"]) >= 2 else lag1_qty
        
        lag1_res = hist["resistances"][-1] if len(hist["resistances"]) >= 1 else r["Resistance_Rate"]
        lag2_res = hist["resistances"][-2] if len(hist["resistances"]) >= 2 else lag1_res
        
        hist["quantities"].append(r["Quantity_Sold"])
        hist["resistances"].append(r["Resistance_Rate"])
        
        recent_q = hist["quantities"][-3:]
        recent_r = hist["resistances"][-3:]
        rolling_3m_qty = float(np.mean(recent_q))
        rolling_3m_res = float(np.mean(recent_r))
        
        item = {
            **r,
            "Quantity_per_1000": qty_per_1000,
            "Antibiotic_Share": antibiotic_share,
            "Total_Monthly_Quantity": tot_monthly_qty,
            "Total_Monthly_DDD": lm_data["Total_Monthly_DDD"],
            "DDD_per_1000_per_day": lm_data["DDD_per_1000_per_day"],
            "Antibiotic_Class_Shift_Percent": lm_data["Antibiotic_Class_Shift_Percent"],
            "Combination_Irrationality_Index_Percent": lm_data["Combination_Irrationality_Index_Percent"],
            "Volatility_Percent": loc_volatility.get(loc, 0.0),
            "Lag1_Quantity_Sold": lag1_qty,
            "Lag2_Quantity_Sold": lag2_qty,
            "Lag1_Resistance_Rate": lag1_res,
            "Lag2_Resistance_Rate": lag2_res,
            "Rolling_3M_Quantity": rolling_3m_qty,
            "Rolling_3M_Resistance": rolling_3m_res
        }
        records.append(item)
        
    return records

# ==============================================================================
# 2. ISOLATION FOREST ANOMALY DETECTOR (Canonical Vectorized Implementation)
# ==============================================================================

class IsolationTreeNode:
    def __init__(self, left=None, right=None, split_feature=None, split_val=None, size=None):
        self.left = left
        self.right = right
        self.split_feature = split_feature
        self.split_val = split_val
        self.size = size
        self.is_leaf = left is None and right is None

def c_factor(n):
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler_gamma = 0.5772156649
    return 2.0 * (math.log(n - 1) + euler_gamma) - (2.0 * (n - 1) / n)

class PureIsolationTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, current_depth=0):
        n_samples, n_features = X.shape
        if current_depth >= self.max_depth or n_samples <= 1:
            return IsolationTreeNode(size=n_samples)
            
        feature_idx = np.random.randint(0, n_features)
        feat_vals = X[:, feature_idx]
        min_v, max_v = feat_vals.min(), feat_vals.max()
        
        if min_v == max_v:
            return IsolationTreeNode(size=n_samples)
            
        split_val = np.random.uniform(min_v, max_v)
        left_mask = feat_vals < split_val
        right_mask = ~left_mask
        
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return IsolationTreeNode(size=n_samples)
            
        left_child = self.fit(X[left_mask], current_depth + 1)
        right_child = self.fit(X[right_mask], current_depth + 1)
        
        return IsolationTreeNode(
            left=left_child,
            right=right_child,
            split_feature=feature_idx,
            split_val=split_val,
            size=n_samples
        )

    def path_length(self, x, node, current_depth=0):
        if node.is_leaf:
            return current_depth + c_factor(node.size)
        if x[node.split_feature] < node.split_val:
            return self.path_length(x, node.left, current_depth + 1)
        else:
            return self.path_length(x, node.right, current_depth + 1)

class FastIsolationForest:
    def __init__(self, n_estimators=100, max_samples=256, contamination=0.08, random_state=42):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        self.trees = []
        self.c_val = None

    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples = len(X)
        sample_size = min(self.max_samples, n_samples)
        max_depth = int(math.ceil(math.log2(max(sample_size, 2))))
        self.c_val = c_factor(sample_size)
        
        self.trees = []
        for _ in range(self.n_estimators):
            sub_indices = np.random.choice(n_samples, sample_size, replace=False)
            X_sub = X[sub_indices]
            tree = PureIsolationTree(max_depth=max_depth)
            tree.root = tree.fit(X_sub)
            self.trees.append(tree)
        return self

    def score_samples(self, X):
        n_samples = len(X)
        paths = np.zeros(n_samples)
        for tree in self.trees:
            for i in range(n_samples):
                paths[i] += tree.path_length(X[i], tree.root)
        avg_paths = paths / len(self.trees)
        # Score s(x, n) = 2^(-E(h(x))/c(n))
        scores = 2.0 ** (- (avg_paths / self.c_val))
        return scores

# ==============================================================================
# 3. GRADIENT BOOSTED REGRESSION ENSEMBLE (XGBoost Regressor Framework)
# ==============================================================================

class FastDecisionTreeRegressor:
    def __init__(self, max_depth=6, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.feature_idx = None
        self.threshold = None
        self.val = None
        self.left = None
        self.right = None

    def fit(self, X, y, depth=0):
        self.val = float(np.mean(y))
        if depth >= self.max_depth or len(y) < self.min_samples_split:
            return
            
        n_samples, n_features = X.shape
        best_loss = float("inf")
        best_f = None
        best_t = None
        
        # Subsample features for variance reduction and speed
        n_sub_feat = max(3, int(n_features * 0.8))
        feat_subset = np.random.choice(n_features, n_sub_feat, replace=False)
        
        var_y = np.var(y) * n_samples
        for f in feat_subset:
            col = X[:, f]
            # Fast quantile thresholds
            thresholds = np.percentile(col, [20, 40, 60, 80])
            for t in thresholds:
                mask = col <= t
                n_l = mask.sum()
                n_r = n_samples - n_l
                if n_l < 4 or n_r < 4:
                    continue
                y_l = y[mask]
                y_r = y[~mask]
                loss = np.var(y_l)*n_l + np.var(y_r)*n_r
                if loss < best_loss:
                    best_loss = loss
                    best_f = f
                    best_t = t
                    
        if best_f is not None and best_loss < var_y * 0.999:
            self.feature_idx = best_f
            self.threshold = best_t
            mask = X[:, best_f] <= best_t
            self.left = FastDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.left.fit(X[mask], y[mask], depth + 1)
            self.right = FastDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            self.right.fit(X[~mask], y[~mask], depth + 1)

    def predict_one(self, x):
        if self.feature_idx is None or self.left is None:
            return self.val
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict_one(x)
        else:
            return self.right.predict_one(x)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

class FastGradientBoostedRegressor:
    def __init__(self, n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []
        self.base_val = 0.0
        self.feature_importances_ = None

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.base_val = float(np.mean(y))
        y_pred = np.full(len(y), self.base_val)
        self.trees = []
        
        n_features = X.shape[1]
        feat_counts = np.zeros(n_features)
        
        for i in range(self.n_estimators):
            residual = y - y_pred
            # Gradient step with shrinkage
            tree = FastDecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=8)
            # Row subsampling (85%)
            sub_mask = np.random.rand(len(X)) < 0.85
            tree.fit(X[sub_mask], residual[sub_mask])
            
            update = tree.predict(X)
            y_pred += self.learning_rate * update
            self.trees.append(tree)
            
            # Count splits for importance
            def count_splits(node):
                if node and node.feature_idx is not None:
                    feat_counts[node.feature_idx] += 1
                    count_splits(node.left)
                    count_splits(node.right)
            count_splits(tree)
            
        tot_splits = feat_counts.sum()
        self.feature_importances_ = feat_counts / tot_splits if tot_splits > 0 else np.ones(n_features)/n_features
        return self

    def predict(self, X):
        y_pred = np.full(len(X), self.base_val)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

# ==============================================================================
# 4. COMPOSITE RISK SCORING
# ==============================================================================

def compute_risk_scores(records):
    """
    Computes composite AMR Risk Score (0-100) & Categorical Stewardship Tier:
    - Resistance Rate Pressure (40%)
    - Isolation Forest Anomaly Score (25%)
    - Combination Irrationality Index (20%)
    - Class Shift & Spectrum Exposure (15%)
    """
    scored = []
    for r in records:
        res_val = min(max(r.get("Predicted_Resistance_Rate", r["Resistance_Rate"]), 0.0), 100.0)
        anomaly_val = min(max(r.get("Anomaly_Risk_Score", 0.0) * 100.0, 0.0), 100.0)
        irrat_val = min(max(r.get("Combination_Irrationality_Index_Percent", 0.0), 0.0), 100.0)
        
        spec_val = SPECTRUM_FACTOR.get(r.get("Spectrum", "Broad"), 50.0)
        shift_val = min(max(r.get("Antibiotic_Class_Shift_Percent", 0.0), 0.0), 100.0)
        spec_shift = (spec_val * 0.6) + (shift_val * 0.4)
        
        risk_score = (
            (res_val * 0.40) +
            (anomaly_val * 0.25) +
            (irrat_val * 0.20) +
            (spec_shift * 0.15)
        )
        risk_score = round(risk_score, 2)
        
        if risk_score >= 75.0:
            tier = "Critical Alert"
            action = "Urgent intervention: Restrict empirical reserve use and audit microbiology culture isolates."
        elif risk_score >= 55.0:
            tier = "High Risk"
            action = "Targeted stewardship review: Investigate high combination & broad-spectrum prescribing."
        elif risk_score >= 30.0:
            tier = "Moderate Risk"
            action = "Elevated vigilance: Track dispensing patterns and monitor seasonal trend shifts."
        else:
            tier = "Low Risk"
            action = "Standard surveillance: Maintain standard stewardship protocols."
            
        scored.append({
            **r,
            "Risk_Score": risk_score,
            "Risk_Tier": tier,
            "Recommended_Action": action
        })
    return scored

# ==============================================================================
# 5. DIAGNOSTICS, PLOTS & REPORTING
# ==============================================================================

def generate_svg_charts(y_true, y_pred, fi_list, records, out_dir):
    """Generates rich standalone SVG visualization charts for reports."""
    
    # 1. Regression Scatter & Identity Line
    w, h = 700, 450
    svg1 = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#ffffff;font-family:Arial,sans-serif;">']
    svg1.append('<text x="350" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">XGBoost: Predicted vs Actual Resistance Rates (%)</text>')
    
    # Bounds
    min_v, max_v = 15.0, 35.0
    def scale_x(v): return 70 + (v - min_v) / (max_v - min_v) * 560
    def scale_y(v): return 380 - (v - min_v) / (max_v - min_v) * 320
    
    # Axes
    svg1.append(f'<line x1="70" y1="380" x2="630" y2="380" stroke="#94a3b8" stroke-width="1.5"/>')
    svg1.append(f'<line x1="70" y1="60" x2="70" y2="380" stroke="#94a3b8" stroke-width="1.5"/>')
    
    # Identity line (y=x)
    svg1.append(f'<line x1="{scale_x(min_v)}" y1="{scale_y(min_v)}" x2="{scale_x(max_v)}" y2="{scale_y(max_v)}" stroke="#ef4444" stroke-width="2" stroke-dasharray="5,5"/>')
    
    # Scatter points (sample 400 for speed & clarity)
    step = max(1, len(y_true) // 400)
    for i in range(0, len(y_true), step):
        cx = scale_x(y_true[i])
        cy = scale_y(y_pred[i])
        svg1.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3" fill="#3b82f6" opacity="0.6"/>')
        
    svg1.append(f'<text x="350" y="415" text-anchor="middle" font-size="12" fill="#475569">Actual Resistance Rate (%)</text>')
    svg1.append(f'<text x="25" y="220" text-anchor="middle" font-size="12" fill="#475569" transform="rotate(-90 25 220)">Predicted Resistance Rate (%)</text>')
    svg1.append('</svg>')
    
    with open(os.path.join(out_dir, "predicted_vs_actual_scatter.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(svg1))
        
    # 2. Top Feature Importances Bar Chart
    svg2 = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" style="background:#ffffff;font-family:Arial,sans-serif;">']
    svg2.append('<text x="350" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">Top Influential Features Driving Resistance Prediction</text>')
    
    top_10 = fi_list[:10]
    max_imp = max(x[1] for x in top_10) if top_10 else 1.0
    for idx, (feat, imp) in enumerate(top_10):
        y_pos = 65 + idx * 32
        bar_len = (imp / max_imp) * 380
        svg2.append(f'<text x="210" y="{y_pos+16}" text-anchor="end" font-size="11" fill="#334155">{feat}</text>')
        svg2.append(f'<rect x="220" y="{y_pos+3}" width="{bar_len:.1f}" height="18" rx="3" fill="#1d4ed8"/>')
        svg2.append(f'<text x="{225+bar_len:.1f}" y="{y_pos+16}" font-size="11" fill="#0f172a" font-weight="bold">{imp:.3f}</text>')
        
    svg2.append('</svg>')
    with open(os.path.join(out_dir, "feature_importance_ranking.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(svg2))
        
    # 3. AMR Risk Tier Distribution Chart
    tier_counts = {}
    for r in records:
        t = r["Risk_Tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
        
    svg3 = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="400" style="background:#ffffff;font-family:Arial,sans-serif;">']
    svg3.append('<text x="350" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#1e293b">AMR Risk Tier Distribution Across Surveillance Records</text>')
    
    tiers = ["Low Risk", "Moderate Risk", "High Risk", "Critical Alert"]
    colors = ["#10b981", "#f59e0b", "#f97316", "#ef4444"]
    max_c = max(tier_counts.values()) if tier_counts else 1
    
    for idx, t in enumerate(tiers):
        c = tier_counts.get(t, 0)
        pct = (c / len(records)) * 100.0
        x_pos = 90 + idx * 140
        b_height = (c / max_c) * 220
        y_pos = 320 - b_height
        
        svg3.append(f'<rect x="{x_pos}" y="{y_pos:.1f}" width="90" height="{b_height:.1f}" rx="4" fill="{colors[idx]}"/>')
        svg3.append(f'<text x="{x_pos+45}" y="{y_pos-8:.1f}" text-anchor="middle" font-size="11" font-weight="bold" fill="#1e293b">{c} ({pct:.1f}%)</text>')
        svg3.append(f'<text x="{x_pos+45}" y="345" text-anchor="middle" font-size="12" fill="#334155">{t}</text>')
        
    svg3.append('</svg>')
    with open(os.path.join(out_dir, "amr_risk_tier_distribution.svg"), "w", encoding="utf-8") as f:
        f.write("\n".join(svg3))

# ==============================================================================
# 6. MAIN PIPELINE RUNNER
# ==============================================================================

def run_amr_pipeline():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, "data")
    results_dir = os.path.join(project_dir, "results")
    eval_dir = os.path.join(results_dir, "evaluation")
    
    for d in [data_dir, results_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)
        
    sales_file = os.path.join(project_dir, "pharmacy_sales.csv")
    resistance_file = os.path.join(project_dir, "resistance_data.csv")
    
    print("\n" + "="*75)
    print("       ANTIMICROBIAL RESISTANCE (AMR) MACHINE LEARNING PIPELINE")
    print("       Isolation Forest + XGBoost Regressor + Risk Score Engine")
    print("="*75)

    # 1. Feature Engineering
    print("\n[STEP 1/5] Running Feature Engineering...")
    records = load_and_engineer_features(sales_file, resistance_file)
    print(f"  [OK] Processed {len(records)} granular antibiotic usage & resistance observations.")
    
    # Save processed features CSV
    feat_keys = list(records[0].keys())
    with open(os.path.join(data_dir, "processed_features.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=feat_keys)
        writer.writeheader()
        writer.writerows(records)

    # 2. Isolation Forest Anomaly Detection
    print("\n[STEP 2/5] Training Isolation Forest Anomaly Detector...")
    iso_feat_keys = [
        "Quantity_Sold", "Quantity_per_1000", "Total_DDD", "Total_Monthly_DDD",
        "DDD_per_1000_per_day", "Antibiotic_Class_Shift_Percent",
        "Combination_Irrationality_Index_Percent", "Volatility_Percent",
        "Antibiotic_Share", "Spectrum_Weight"
    ]
    
    X_iso = np.array([[r[k] for k in iso_feat_keys] for r in records], dtype=np.float64)
    iso_forest = FastIsolationForest(n_estimators=100, max_samples=256, contamination=0.08, random_state=42)
    iso_forest.fit(X_iso)
    raw_anomaly_scores = iso_forest.score_samples(X_iso)
    
    # Invert/normalize scores so 1.0 is highest anomaly risk
    min_s, max_s = raw_anomaly_scores.min(), raw_anomaly_scores.max()
    normalized_anomaly_scores = (raw_anomaly_scores - min_s) / (max_s - min_s) if max_s > min_s else np.zeros_like(raw_anomaly_scores)
    threshold_92 = float(np.percentile(normalized_anomaly_scores, 92))
    
    anomaly_flags = (normalized_anomaly_scores >= threshold_92).astype(int)
    for i, r in enumerate(records):
        r["Anomaly_Risk_Score"] = round(float(normalized_anomaly_scores[i]), 4)
        r["Anomaly_Flag"] = int(anomaly_flags[i])
        
    print(f"  [OK] Isolation Forest trained on {len(iso_feat_keys)} multidimensional usage features.")
    print(f"  [OK] Identified {anomaly_flags.sum()} anomalous usage patterns ({anomaly_flags.mean()*100:.2f}%).")

    # 3. XGBoost Resistance Regressor Training & Temporal Validation
    print("\n[STEP 3/5] Training XGBoost Resistance Regressor (Chronological Partition)...")
    xgb_feat_keys = [
        "Population", "Latitude", "Longitude", "Quantity_Sold", "Total_DDD",
        "Quantity_per_1000", "Total_Monthly_DDD", "DDD_per_1000_per_day",
        "Antibiotic_Share", "Spectrum_Weight", "Is_Combination",
        "Antibiotic_Class_Shift_Percent", "Combination_Irrationality_Index_Percent",
        "Volatility_Percent", "Lag1_Quantity_Sold", "Lag2_Quantity_Sold",
        "Lag1_Resistance_Rate", "Lag2_Resistance_Rate", "Rolling_3M_Quantity",
        "Rolling_3M_Resistance", "Month_Number", "Quarter",
        "Anomaly_Flag", "Anomaly_Risk_Score"
    ]
    
    # Antibiotic and Class integer encoding
    all_antibiotics = sorted(list(set(r["Antibiotic"] for r in records)))
    all_classes = sorted(list(set(r["Class"] for r in records)))
    anti_map = {a: i for i, a in enumerate(all_antibiotics)}
    class_map = {c: i for i, c in enumerate(all_classes)}
    
    X_all = []
    y_all = []
    for r in records:
        row = [float(r[k]) for k in xgb_feat_keys]
        row.append(anti_map[r["Antibiotic"]])
        row.append(class_map[r["Class"]])
        X_all.append(row)
        y_all.append(float(r["Resistance_Rate"]))
        
    X_all = np.array(X_all, dtype=np.float64)
    y_all = np.array(y_all, dtype=np.float64)
    
    # Sort chronologically by Month string
    month_order = sorted(list(set(r["Month"] for r in records)))
    split_month_idx = int(len(month_order) * 0.8)
    train_months = set(month_order[:split_month_idx])
    test_months = set(month_order[split_month_idx:])
    
    train_mask = np.array([r["Month"] in train_months for r in records])
    test_mask = ~train_mask
    
    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test, y_test = X_all[test_mask], y_all[test_mask]
    
    print(f"  -> Training set size: {len(X_train)} | Out-of-time test set size: {len(X_test)}")
    xgb_reg = FastGradientBoostedRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42)
    xgb_reg.fit(X_train, y_train)
    
    test_preds = xgb_reg.predict(X_test)
    full_preds = xgb_reg.predict(X_all)
    
    # Metrics
    test_r2 = 1.0 - (np.sum((y_test - test_preds)**2) / np.sum((y_test - np.mean(y_test))**2))
    test_rmse = float(np.sqrt(np.mean((y_test - test_preds)**2)))
    test_mae = float(np.mean(np.abs(y_test - test_preds)))
    test_mape = float(np.mean(np.abs((y_test - test_preds) / np.clip(y_test, 1e-5, None))) * 100.0)
    
    for i, r in enumerate(records):
        r["Predicted_Resistance_Rate"] = round(float(full_preds[i]), 2)
        
    all_feat_names = xgb_feat_keys + ["Antibiotic_Code", "Class_Code"]
    fi_list = sorted(list(zip(all_feat_names, xgb_reg.feature_importances_)), key=lambda x: x[1], reverse=True)
    
    print("  [OK] XGBoost Model Evaluation Results:")
    print(f"    - Test R^2 Score : {test_r2:.4f} (Model captures {test_r2*100:.2f}% of variance)")
    print(f"    - Test RMSE     : {test_rmse:.2f}%")
    print(f"    - Test MAE      : {test_mae:.2f}%")
    print(f"    - Test MAPE     : {test_mape:.2f}%")

    # 4. Composite Risk Scoring
    print("\n[STEP 4/5] Computing Multi-Factor AMR Risk Scores...")
    scored_records = compute_risk_scores(records)
    
    # Save complete predictions and risks
    with open(os.path.join(results_dir, "amr_complete_predictions_and_risks.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored_records[0].keys()))
        writer.writeheader()
        writer.writerows(scored_records)
        
    # Location Risk Summary
    loc_summary = {}
    for r in scored_records:
        loc = r["Location"]
        if loc not in loc_summary:
            loc_summary[loc] = {"Risk_Scores": [], "Criticals": 0, "Highs": 0, "Resistances": []}
        loc_summary[loc]["Risk_Scores"].append(r["Risk_Score"])
        loc_summary[loc]["Resistances"].append(r["Predicted_Resistance_Rate"])
        if r["Risk_Tier"] == "Critical Alert":
            loc_summary[loc]["Criticals"] += 1
        elif r["Risk_Tier"] == "High Risk":
            loc_summary[loc]["Highs"] += 1
            
    summary_rows = []
    for loc, data in loc_summary.items():
        summary_rows.append({
            "Location": loc,
            "Avg_Risk_Score": round(float(np.mean(data["Risk_Scores"])), 2),
            "Max_Risk_Score": round(float(np.max(data["Risk_Scores"])), 2),
            "Critical_Alerts_Count": data["Criticals"],
            "High_Risk_Count": data["Highs"],
            "Avg_Predicted_Resistance": round(float(np.mean(data["Resistances"])), 2)
        })
    summary_rows.sort(key=lambda x: x["Avg_Risk_Score"], reverse=True)
    
    with open(os.path.join(results_dir, "location_risk_ranking.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # 5. Visualizations & Evaluation Report
    print("\n[STEP 5/5] Generating Visual Evaluation Suite & Reports...")
    generate_svg_charts(y_test, test_preds, fi_list, scored_records, eval_dir)
    
    # Markdown Report
    tier_counts = {}
    for r in scored_records:
        t = r["Risk_Tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
        
    report_md = f"""# Antimicrobial Resistance (AMR) Machine Learning Evaluation Report

## 1. Executive Model Performance Summary
An end-to-end Machine Learning pipeline was constructed combining **Isolation Forest Anomaly Detection**, **XGBoost Resistance Prediction**, and **Composite AMR Risk Scoring**.

### XGBoost Model Diagnostics (Out-of-Time Forward Test Split)
- **R² Determination Coefficient**: `{test_r2:.4f}`
- **Root Mean Squared Error (RMSE)**: `{test_rmse:.2f}%`
- **Mean Absolute Error (MAE)**: `{test_mae:.2f}%`
- **Mean Absolute Percentage Error (MAPE)**: `{test_mape:.2f}%`

---

## 2. Top 10 Most Influential Features (XGBoost Importance)
"""
    for idx, (feat, imp) in enumerate(fi_list[:10]):
        report_md += f"{idx+1}. **{feat}**: `{imp:.4f}`\n"

    report_md += f"""
---

## 3. Isolation Forest Anomaly Detection Summary
- **Contamination Target**: 8%
- **Anomalous Usage Patterns Identified**: {anomaly_flags.sum()} ({anomaly_flags.mean()*100:.2f}%)
- **Baseline Patterns**: {len(records) - anomaly_flags.sum()} ({(1-anomaly_flags.mean())*100:.2f}%)

---

## 4. Multi-Factor AMR Risk Tier Breakdown
- **Critical Alert (≥ 75)**: {tier_counts.get('Critical Alert', 0)} ({tier_counts.get('Critical Alert', 0)/len(records)*100:.1f}%)
- **High Risk (55–74.9)**: {tier_counts.get('High Risk', 0)} ({tier_counts.get('High Risk', 0)/len(records)*100:.1f}%)
- **Moderate Risk (30–54.9)**: {tier_counts.get('Moderate Risk', 0)} ({tier_counts.get('Moderate Risk', 0)/len(records)*100:.1f}%)
- **Low Risk (< 30)**: {tier_counts.get('Low Risk', 0)} ({tier_counts.get('Low Risk', 0)/len(records)*100:.1f}%)

---

## 5. Top Highest-Risk Locations
"""
    for row in summary_rows[:5]:
        report_md += f"- **{row['Location']}**: Avg Risk Score = `{row['Avg_Risk_Score']}`, Critical Alerts = `{row['Critical_Alerts_Count']}`\n"

    report_path = os.path.join(eval_dir, "model_evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n" + "="*75)
    print("             PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*75)
    print(f"• Processed Data    : {data_dir}")
    print(f"• Risk Predictions  : {results_dir}")
    print(f"• Evaluation Charts : {eval_dir}")
    print(f"• Evaluation Report : {report_path}")
    print("="*75 + "\n")

if __name__ == "__main__":
    run_amr_pipeline()
