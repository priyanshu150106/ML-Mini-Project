# Unit 2: Advertisement Click Prediction

## Overview
This unit builds an interpretable binary classification system to predict whether a website visitor will click on an online digital advertisement based on user demographics and engagement history.

## Key Links
- **Google Colab Notebook:** [Colab Implementation](https://colab.research.google.com/drive/1ZQLfjucPS5IzBeQduiwBqBOjK8kLYam5?usp=sharing)
- **Dataset:** [Kaggle Advertising Dataset](https://www.kaggle.com/datasets/abhinavgautam/advertising/data)
- **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1qyjvYdyOY2Se6DihbYU0HQPRom9LxRbw/edit?usp=drive_link&ouid=111278213256287944154&rtpof=true&sd=true)
- **Unit Report:** [Google Drive PDF](https://drive.google.com/file/d/1dk-MmbkNZNwkGjCrc4AJPMW54zJ6Yw18/view?usp=sharing)

## Problem Formulation & Model
- **Supervised Model:** Binary Logistic Regression with `StandardScaler` data normalization.
- **Predictors:** `Age`, `Area Income`, `Daily Time Spent on Site`, `Daily Internet Usage`, `Male`.
- **Target:** `Clicked on Ad` (0 = No Click, 1 = Click).

## Performance Summary (200 Test Visitors)
- **Accuracy:** 96.0% (192 / 200 correct predictions)
- **ROC-AUC Score:** 0.98
- **Confusion Matrix:**
  - True Negatives ($TN$): 86
  - False Positives ($FP$): 3
  - False Negatives ($FN$): 5
  - True Positives ($TP$): 106
- **Balanced Class F1:** 0.96 for both classes

## Learned Standardized Feature Coefficients
| Feature | Standardized $\beta$ | Interpretation |
|---|---|---|
| `Age` | +1.2587 | Older users show higher click propensity |
| `Male` | -0.3587 | Minimal direct gender effect |
| `Area Income` | -1.6336 | Higher income visitors click less frequently |
| `Daily Time Spent on Site` | -2.5332 | Power visitors ignore display banners |
| `Daily Internet Usage` | -2.6987 | Heavy internet consumers exhibit banner blindness |

## How to Run
```bash
python ad_click_prediction.py
```
Or open `Unit_2_Ad_Click_Prediction.ipynb` in Jupyter Notebook / Google Colab.
