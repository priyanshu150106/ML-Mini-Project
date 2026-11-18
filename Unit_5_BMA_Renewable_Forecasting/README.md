# Unit 5: Renewable Forecasting & Battery Dispatch via Bayesian Model Averaging

## Overview
This unit formulates a rolling Bayesian Model Averaging (BMA) ensemble framework to combine diverse day-ahead wind and solar generation forecasts in Germany (OPSD, 2015-2019), quantify aggregate power uncertainty via a Gaussian copula, and guide cost-effective energy commitments with battery storage.

## Key Links
- **Google Colab Notebook:** [Colab Implementation](https://colab.research.google.com/drive/1rxozy93DZf-Te1kDaeszMibY_P9cVio0?usp=sharing)
- **Dataset:** [Open Power System Data (OPSD) Time Series](https://data.open-power-system-data.org/time_series/)
- **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1q3EaaWO-0cwhWasm_YJNTrxsoYcu13Df/edit?usp=sharing&ouid=103491267144499532443&rtpof=true&sd=true)
- **Unit Report:** [Google Drive PDF](https://drive.google.com/file/d/1L4nI8YPlvPMmnPAP1009QYJh5OC3R_ua/view?usp=sharing)

## Methodology & Formulation
- **Predictive Mixture Density:**
  $$p(y \mid f_1, \dots, f_K) = \sum_{k=1}^K w_k \cdot \mathcal{N}\left(a_k + b_k f_k, \sigma^2\right)$$
  - Linear member bias calibration ($a_k, b_k$) over a 40-day sliding window.
  - EM algorithm for dynamically learning model responsibility weights $w_k$ and shared residual error variance $\sigma^2$.
- **Ensemble Members (Day-Ahead):**
  1. `Persistence_24h`
  2. `Mean_7d` (diurnal baseline)
  3. `Ridge` (L2 regularized linear regression)
  4. `RandomForest` (bagged decision trees)
  5. `GradBoost` (gradient boosted trees)
  6. `GradBoost_weather_only` (weather feature specialist)
- **Joint Uncertainty:** Parametric Gaussian copula linking marginal wind and solar distributions ($\rho = 0.031$).
- **Storage Facility Simulation:** 60 MW wind + 40 MW solar coupled with a 20 MW / 80 MWh battery ($\eta = 0.95$, SOC: 10%–90%).
  - Pricing: Shortage penalty = 120 EUR/MWh, Spillage loss = 30 EUR/MWh, Degradation cost = 5 EUR/MWh throughput.

## 2019 Held-Out Forecasting Benchmarks
| Target | Model | MAE | RMSE | CRPS | 90% Coverage | 90% Width |
|---|---|---|---|---|---|---|
| **Wind** | Equal-Weight Ensemble | 0.0759 | 0.0927 | 0.0529 | 0.976 | 0.411 |
| Wind | **Rolling BMA** | **0.0617** | **0.0783** | **0.0439** | **0.880** | **0.240** |
| **Solar** | Equal-Weight Ensemble | 0.0243 | 0.0361 | 0.0196 | 0.972 | 0.158 |
| Solar | **Rolling BMA** | **0.0211** | **0.0315** | **0.0161** | **0.885** | **0.088** |

## Battery Simulation Operating Costs (2019 Test Year)
| Strategy | Total Cost (kEUR) | Shortage (MWh) | Spilled (MWh) | Battery Cycles | Savings vs Best Single |
|---|---|---|---|---|---|
| Best Single (Random Forest) + Battery | 2,687.3 | 20,753.3 | 4,424.3 | 80.2 | Baseline |
| Equal-Weight Mean + Battery | 2,296.5 | 15,167.0 | 14,039.0 | 69.2 | 14.5% |
| BMA Mean + Battery | 1,641.6 | 10,683.0 | 9,845.6 | 80.3 | 38.9% |
| BMA Quantile ($\tau = 0.28$) + Battery | 1,107.4 | 2,568.0 | 24,938.3 | 63.9 | 58.8% |
| **Adaptive BMA ($\tau_0 = 0.26, \kappa = 0.6$) + Battery** | **504.3** | **249.1** | **13,461.8** | **88.2** | **81.2%** |

## How to Run
```bash
python bma_renewable_battery.py
```
Or open `Unit_5_BMA_Forecasting.ipynb` in Jupyter Notebook / Google Colab.
