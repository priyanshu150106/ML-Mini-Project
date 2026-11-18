# Unit 1: Predictive Pothole Risk Detection

## Overview
This unit investigates proactive road infrastructure maintenance by predicting whether a road segment will develop potholes within a 90-day forward-looking window.

## Key Links
- **Google Colab Notebook:** [Colab Implementation](https://colab.research.google.com/drive/1vdnxaDtFWs3nS3mXF-cxtu_Zd0WCeiUl?usp=sharing)
- **Dataset:** [Zenodo Record 15638682](https://zenodo.org/records/15638682) (3,000 road sections)
- **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1Iu8aryOgr1Jc8gYrUsUqy-s3YyqagVlh/edit?usp=drivesdk&ouid=103491267144499532443&rtpof=true&sd=true)
- **Unit Report:** [Google Docs Report](https://docs.google.com/document/d/1IIVtdY49cWUrw07Oku0g4jNRtEbe4PXu/edit?usp=drivesdk&ouid=103491267144499532443&rtpof=true&sd=true)

## Problem Formulation & Models
- **Supervised Task:** Binary classification on 14 engineered features.
- **Physical Feature Engineering:**
  - `traffic_load_index` $= \text{volume} \times (1 + 2 \times \text{heavy\%} / 100)$
  - `maint_delay_ratio` $= \text{days\_since\_maint} / 365$
  - `wetness_stress` $= \text{rainfall} \times \text{temp\_variation} / 100$
- **Algorithms Compared:**
  1. Standardized Logistic Regression (`class_weight='balanced'`)
  2. Decision Tree Classifier (`max_depth=5, min_samples_leaf=20`)
  3. Random Forest Classifier (300 estimators, `max_depth=8`)
  4. XGBoost Classifier (300 estimators, `max_depth=3`, $\eta=0.05$)

## Quantitative Results (Held-Out Test Set)
| Model | Accuracy | Precision | Recall (Priority) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.737** | **0.605** | **0.714** | **0.655** | **0.806** |
| Decision Tree | 0.655 | 0.507 | 0.490 | 0.499 | 0.683 |
| Random Forest | 0.740 | 0.638 | 0.595 | 0.616 | 0.795 |
| XGBoost | 0.758 | 0.721 | 0.505 | 0.594 | 0.801 |

## Risk Stratification Tiers
- **Low Risk ($p < 0.33$):** 12.0% empirical pothole rate $\rightarrow$ Routine monitoring.
- **Medium Risk ($0.33 \le p < 0.66$):** 37.2% empirical pothole rate $\rightarrow$ Inspection within 30 days.
- **High Risk ($p \ge 0.66$):** 69.9% empirical pothole rate $\rightarrow$ Preventive resurfacing.

## How to Run
```bash
python pothole_prediction.py
```
Or open `Unit_1_Pothole_Risk_Prediction.ipynb` in Jupyter Notebook / Google Colab.
