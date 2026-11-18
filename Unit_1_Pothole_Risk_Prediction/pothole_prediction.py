"""
Unit 1: Predictive Pothole Risk Detection
Part of 21CSC305P Machine Learning Mini Project
Academic Year: 2026-2027 | SRM Institute of Science and Technology

Problem Statement:
Predict whether a road section will develop potholes within 90 days using
road attributes, environmental factors, traffic load, and maintenance history,
and categorize each section into Low, Medium, or High risk for preventive maintenance.

Models compared:
- Logistic Regression (standardized, class-balanced)
- Decision Tree (CART)
- Random Forest Classifier (ensemble of 300 trees)
- XGBoost Classifier (gradient boosting)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

def run_unit_1(output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(42)
    n = 3000

    print("=" * 70)
    print("STEP 1: Generating Synthetic Road-Section Dataset (N = 3,000)")
    print("=" * 70)

    df = pd.DataFrame({
        "section_id": [f"R{100+i}" for i in range(n)],
        "road_age": rng.integers(1, 41, n),
        "surface_material": rng.choice(["Asphalt", "Concrete", "Chip-seal"], n, p=[.6, .25, .15]),
        "segment_length_km": rng.uniform(0.2, 5, n).round(2),
        "traffic_volume": rng.lognormal(8.3, 0.7, n).round(0),
        "heavy_vehicle_pct": rng.beta(2, 8, n).round(3) * 100,
        "rainfall_mm_90d": rng.gamma(4, 55, n).round(1),
        "temp_variation_c": rng.normal(14, 4, n).clip(3, 30).round(1),
        "condition_score": rng.normal(60, 18, n).clip(5, 100).round(1),
        "crack_density": rng.beta(2, 6, n).round(3),
        "days_since_maint": rng.integers(30, 2500, n),
        "maint_per_year": rng.poisson(0.8, n),
        "prior_potholes": rng.poisson(1.2, n),
    })

    df["condition_change"] = (-0.04 * df.road_age - 8 * df.crack_density + rng.normal(0, 1.5, n)).round(2)
    z = lambda s: (s - s.mean()) / s.std()
    mat = df.surface_material.map({"Asphalt": 0.0, "Concrete": -0.5, "Chip-seal": 0.3})

    logit = (
        -0.9
        + 0.55 * z(df.road_age)
        + 0.45 * z(np.log(df.traffic_volume))
        + 0.30 * z(df.heavy_vehicle_pct)
        + 0.50 * z(df.rainfall_mm_90d)
        + 0.30 * z(df.temp_variation_c)
        - 0.65 * z(df.condition_score)
        + 0.35 * z(df.crack_density)
        + 0.40 * z(df.days_since_maint)
        - 0.25 * df.maint_per_year
        + 0.30 * df.prior_potholes
        + mat
        + 0.35 * z(df.rainfall_mm_90d) * z(np.log(df.traffic_volume))   # interaction
        + 0.30 * z(df.road_age) * (-z(df.condition_score))             # interaction
    )
    p = 1 / (1 + np.exp(-logit))
    df["pothole_next_90d"] = (rng.random(n) < p).astype(int)

    # Inject realistic defects (missing values and outliers)
    for col, frac in [("rainfall_mm_90d", .04), ("condition_score", .03), ("traffic_volume", .03), ("days_since_maint", .05)]:
        df.loc[rng.choice(n, int(frac * n), replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "traffic_volume"] *= 8

    raw_csv = os.path.join(output_dir, "pothole_synthetic_raw.csv")
    df.to_csv(raw_csv, index=False)
    print(f"Dataset generated. Saved to: {raw_csv}")
    print(f"Pothole positive rate: {df.pothole_next_90d.mean():.1%}")

    print("\n" + "=" * 70)
    print("STEP 2: Data Cleaning and Outlier Handling")
    print("=" * 70)
    num_cols = [
        "road_age", "segment_length_km", "traffic_volume", "heavy_vehicle_pct",
        "rainfall_mm_90d", "temp_variation_c", "condition_score", "crack_density",
        "days_since_maint", "maint_per_year", "prior_potholes"
    ]
    clean = df.copy()
    clean[num_cols] = clean[num_cols].fillna(clean[num_cols].median())
    q1, q3 = clean.traffic_volume.quantile([.25, .75])
    iqr = q3 - q1
    hi = q3 + 3 * iqr
    clipped_outliers = int((clean.traffic_volume > hi).sum())
    clean["traffic_volume"] = clean.traffic_volume.clip(upper=hi)
    print(f"Imputed missing columns with median. Clipped {clipped_outliers} extreme traffic outliers (fence: {hi:.1f}).")

    print("\n" + "=" * 70)
    print("STEP 3: Feature Engineering")
    print("=" * 70)
    clean["traffic_load_index"] = clean.traffic_volume * (1 + 2 * clean.heavy_vehicle_pct / 100)
    clean["maint_delay_ratio"] = clean.days_since_maint / 365
    clean["wetness_stress"] = clean.rainfall_mm_90d * clean.temp_variation_c / 100
    clean = pd.concat([
        clean,
        pd.get_dummies(clean.surface_material, prefix="mat", drop_first=True).astype(int)
    ], axis=1)

    features = [
        "road_age", "segment_length_km", "traffic_load_index", "rainfall_mm_90d",
        "temp_variation_c", "condition_score", "crack_density", "maint_delay_ratio",
        "maint_per_year", "prior_potholes", "condition_change", "wetness_stress",
        "mat_Chip-seal", "mat_Concrete"
    ]
    X, y = clean[features], clean.pothole_next_90d

    # Save cleaned and engineered dataset
    clean_csv = os.path.join(output_dir, "pothole_cleaned_features.csv")
    clean.to_csv(clean_csv, index=False)

    print("\n" + "=" * 70)
    print("STEP 4: Train / Test Split & Feature Standardization")
    print("=" * 70)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    print(f"Training set: {len(Xtr)} samples | Test set: {len(Xte)} samples")

    print("\n" + "=" * 70)
    print("STEP 5: Model Training and Evaluation")
    print("=" * 70)
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1), False),
        "XGBoost": (XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42), False),
    }

    res, probs, fitted = {}, {}, {}
    for name, (m, scaled) in models.items():
        X_train_cur = Xtr_s if scaled else Xtr
        X_test_cur = Xte_s if scaled else Xte
        m.fit(X_train_cur, ytr)
        pr = m.predict_proba(X_test_cur)[:, 1]
        pred = (pr >= 0.5).astype(int)
        cm = confusion_matrix(yte, pred)
        res[name] = {
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "AUC": roc_auc_score(yte, pr),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1])
        }
        probs[name] = pr
        fitted[name] = m

    # 5-fold CV
    print("\n--- 5-Fold Stratified Cross-Validation (Full Dataset ROC-AUC) ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = {}
    for name, (m, scaled) in models.items():
        X_cur = scaler.fit_transform(X) if scaled else X
        scores = cross_val_score(m, X_cur, y, cv=cv, scoring="roc_auc")
        cv_scores[name] = (scores.mean(), scores.std())
        print(f"{name:20s}: Mean AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    res_df = pd.DataFrame(res).T
    print("\n--- Test Set Performance Metrics (Decision Threshold = 0.5) ---")
    print(res_df[["Accuracy", "Precision", "Recall", "F1", "AUC"]].round(3))

    best_model_name = max(res, key=lambda k: res[k]["AUC"])
    print(f"\nBest Model by ROC-AUC: {best_model_name} (AUC = {res[best_model_name]['AUC']:.3f})")

    print("\n" + "=" * 70)
    print("STEP 6: Decision Threshold Optimization & Risk Band Assignment")
    print("=" * 70)
    bp = probs[best_model_name]
    threshold_results = []
    for th in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_th = (bp >= th).astype(int)
        threshold_results.append({
            "Threshold": th,
            "Precision": precision_score(yte, pred_th),
            "Recall": recall_score(yte, pred_th),
            "F1": f1_score(yte, pred_th)
        })
    th_df = pd.DataFrame(threshold_results)
    print("Threshold Tuning Analysis:")
    print(th_df.round(3).to_string(index=False))

    # Risk bands
    levels = pd.cut(bp, [0, .33, .66, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
    band_analysis = pd.DataFrame({
        "Count": levels.value_counts(),
        "Observed Pothole Rate": pd.Series(yte.values).groupby(levels.astype(str)).mean()
    })
    print("\nOperational Risk Bands on Test Set:")
    print(band_analysis)

    print("\n" + "=" * 70)
    print("STEP 7: Worked Production Example - Section R-310")
    print("=" * 70)
    ex = pd.DataFrame([dict(
        road_age=12, segment_length_km=2.0, traffic_load_index=9000 * 1.5,
        rainfall_mm_90d=320, temp_variation_c=18, condition_score=30,
        crack_density=0.35, maint_delay_ratio=18 * 30 / 365, maint_per_year=0,
        prior_potholes=3, condition_change=-3.0, wetness_stress=320 * 18 / 100,
        **{"mat_Chip-seal": 0, "mat_Concrete": 0}
    )])[features]

    bm = fitted[best_model_name]
    ex_input = scaler.transform(ex) if models[best_model_name][1] else ex
    p_ex = float(bm.predict_proba(ex_input)[:, 1][0])
    risk_level = "High" if p_ex >= 0.66 else ("Medium" if p_ex >= 0.33 else "Low")
    print(f"Profile: Road Age=12 yrs, Traffic Index=13500, Rain=320mm, Condition Score=30, Prior Potholes=3")
    print(f"Calculated Pothole Probability: {p_ex:.1%}")
    print(f"Assigned Risk Tier: {risk_level} Risk")
    print(f"Recommended Maintenance Action: Immediate preventive resurfacing and drainage inspection.")

    return res

if __name__ == "__main__":
    run_unit_1()
