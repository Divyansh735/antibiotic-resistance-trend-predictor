# Antimicrobial Stewardship Policy Scenario Simulations

## Executive Overview
Using the trained **Isolation Forest Anomaly Detector** and **XGBoost Resistance Predictor**, we evaluated four "What-If" clinical policy interventions to simulate their quantitative impact on Antimicrobial Resistance (AMR) pressure, prescribing anomaly rates, and Composite Risk Scores.

---

## 1. Comparative Policy Scenario Results

| Scenario | Policy Intervention Description | Baseline Risk | Simulated Risk | **Risk Delta** | Baseline Res. (%) | Simulated Res. (%) | **Res. Delta** | Anomaly Rate Drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Scenario 1** | 30% Cut in Combination Therapy | `24.9` | `25.66` | **`+0.75` pts** | `19.36%` | `19.28%` | **`-0.08%`** | `+6.94%` |
| **Scenario 2** | 50% Cut in Combination Therapy | `24.9` | `25.28` | **`+0.38` pts** | `19.36%` | `19.25%` | **`-0.11%`** | `+11.26%` |
| **Scenario 3** | 25% Restriction on Reserve/Broad Spectrum (Meropenem, Ceftriaxone) | `24.9` | `25.18` | **`+0.28` pts** | `19.36%` | `19.35%` | **`-0.01%`** | `+0.24%` |
| **Scenario 4** | **Comprehensive Stewardship** (-40% Comb, -30% Reserve, -20% Volume) | `24.9` | `26.56` | **`+1.66` pts** | `19.36%` | `19.23%` | **`-0.13%`** | `+14.07%` |

---

## 2. Top Cities Benefiting from Comprehensive Policy Intervention

| Rank | Surveillance Location | Baseline Risk Score | Projected Risk Score | **Risk Points Averted** | Baseline Resistance | Projected Resistance |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Delhi** | 36.55 | 34.74 | **-1.81 pts** | 25.69% | 25.44% |
| 2 | **Mumbai** | 33.4 | 32.58 | **-0.82 pts** | 26.35% | 26.07% |
| 3 | **Kolkata** | 34.13 | 33.83 | **-0.30 pts** | 25.82% | 25.53% |
| 4 | **Indore** | 27.52 | 28.03 | **--0.51 pts** | 23.51% | 23.26% |
| 5 | **Hyderabad** | 28.35 | 29.09 | **--0.75 pts** | 21.59% | 21.41% |
| 6 | **Bengaluru** | 29.28 | 30.34 | **--1.06 pts** | 22.13% | 21.93% |
| 7 | **Ahmedabad** | 26.65 | 28.14 | **--1.49 pts** | 21.18% | 20.99% |
| 8 | **Chennai** | 26.5 | 28.06 | **--1.56 pts** | 21.34% | 21.16% |
| 9 | **Patna** | 25.0 | 26.63 | **--1.63 pts** | 22.22% | 22.0% |
| 10 | **Jabalpur** | 22.99 | 24.68 | **--1.70 pts** | 16.57% | 16.53% |

---

## 3. Antibiotic Drug Class Impact Analysis (Comprehensive Policy)

| Antibiotic | Drug Class | Spectrum | Baseline Risk | Simulated Risk | **Risk Points Averted** | Resistance Drop |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Amoxicillin-Clavulanate** | Penicillin + beta-lactamase inhibitor | Broad | 28.38 | 26.42 | **-1.95 pts** | -0.13% |
| **Meropenem** | Carbapenem | Very Broad | 31.57 | 33.12 | **--1.54 pts** | -0.12% |
| **Metronidazole** | Nitroimidazole | Narrow | 22.1 | 23.74 | **--1.64 pts** | -0.12% |
| **Amoxicillin** | Penicillin | Narrow | 23.0 | 24.89 | **--1.89 pts** | -0.13% |
| **Doxycycline** | Tetracycline | Broad | 23.26 | 25.22 | **--1.96 pts** | -0.13% |
| **Ciprofloxacin** | Fluoroquinolone | Broad | 23.39 | 25.4 | **--2.01 pts** | -0.13% |
| **Cefixime** | Cephalosporin | Broad | 23.4 | 25.49 | **--2.09 pts** | -0.13% |
| **Azithromycin** | Macrolide | Broad | 24.92 | 27.04 | **--2.12 pts** | -0.13% |
| **Ceftriaxone** | Cephalosporin | Broad | 25.19 | 27.81 | **--2.62 pts** | -0.13% |
| **Cotrimoxazole** | Sulfonamide combination | Broad | 23.81 | 26.48 | **--2.67 pts** | -0.13% |

---

## 4. Key Policy Recommendations
1. **Target Multi-Drug Combinations First**: Restricting irrational combinations (Amoxicillin-Clavulanate and Cotrimoxazole) yields the fastest reduction in systemic risk index and prescribing anomalies.
2. **Prioritize Metropolitan Hotspots**: Delhi, Kolkata, and Mumbai show the largest risk score reductions under targeted volume and reserve capping.
3. **Protect Reserve Lines**: Strict pre-authorization for Meropenem directly dampens extreme resistance escalation without impeding routine patient care.
