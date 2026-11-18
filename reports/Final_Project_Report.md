# APPLIED MACHINE LEARNING: CLASSIFICATION, CLUSTERING, SEQUENCE MODELLING AND ENSEMBLE FORECASTING

**21CSC305P – MACHINE LEARNING (2021 Regulation)**  
**III Year / V Semester | Academic Year: 2026 - 2027**  

<div align="center">
  <img src="srm_logo.jpg" alt="SRM Logo" width="300"/>
  <h3>DEPARTMENT OF COMPUTING TECHNOLOGIES</h3>
  <h3>SCHOOL OF COMPUTING, COLLEGE OF ENGINEERING AND TECHNOLOGY</h3>
  <h3>SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, KATTANKULATHUR - 603 203</h3>
  <p><strong>NOVEMBER 2026</strong></p>
</div>

---

### Submitted By:
- **Hrichik Ghosh** [Reg. No: RA2411003011784]
- **Namit Kaushik** [Reg. No: RA2411003011795]
- **Priyanshu Raman** [Reg. No: RA2411003011800]

### Under the Guidance of:
- **Dr. Kavi Priya D**, Assistant Professor, Department of Computing Technologies
- **Dr. G. Niranjana**, Professor & Head of the Department, Department of Computing Technologies

---

## BONAFIDE CERTIFICATE

Certified that this **21CSC305P - MACHINE LEARNING** project report is the bonafide work of **"Hrichik Ghosh [RA2411003011784], Namit Kaushik [RA2411003011795], Priyanshu Raman [RA2411003011800]"** who carried out the task of completing the project within the allotted time.

<div align="center">
  <img src="bonafide_signed.jpg" alt="Signed Bonafide Certificate" width="550"/>
  <br/>
  <a href="Signed_Bonafide_Certificate.pdf">📄 <strong>Download Signed Bonafide Certificate (PDF)</strong></a>
</div>

---

## ABSTRACT

This report presents an integrated study of five machine learning techniques implemented across three project-based learning stages. The overall problem was to apply supervised, unsupervised, sequential and ensemble methods to practical prediction and decision-support tasks, and to evaluate each with metrics suited to its data. In PBL 1, Unit 1 predicted road sections likely to develop potholes within 90 days using a synthetic dataset of 3,000 sections, where Logistic Regression gave the best ROC-AUC (0.806) of four classifiers, and Unit 2 predicted advertisement clicks on the 1,000-record Advertising dataset, where Logistic Regression reached 96% accuracy and an AUC of 0.98. In PBL 2, Unit 3 applied K-Means to colour quantization of 24 Kodak images, where 16 colours gave a six-fold smaller representation with a mean PSNR of 30.21 dB and SSIM of 0.881, and Unit 4 used Gaussian Hidden Markov Models with MFCC features on the Google Speech Commands data to detect the wake word “marvin”, reaching 72.8% accuracy with a hand-written Forward algorithm verified against a library. In PBL 3, Unit 5 applied Bayesian Model Averaging to hourly German wind and solar data from the Open Power System Data, reducing the mean absolute error to 0.0617 and 0.0211 against 0.0759 and 0.0243 for equal-weight averaging, while an adaptive quantile strategy gave the lowest simulated battery cost of 504.3 kEUR. The study shows that technique selection, threshold choice and honest evaluation matter as much as model complexity, although the pothole data are synthetic and each experiment used one dataset.

---

## TABLE OF CONTENTS

| CHAPTER NO. | TITLE | PAGE NO. |
|---|---|---|
| **1** | **INTRODUCTION** | 1 |
| | 1.1 Background, Motivation and Scope | 1 |
| | 1.2 Integrated Problem Statement | 1 |
| | 1.3 Project Objectives | 2 |
| | 1.4 Unit-wise Topic and Dataset Allocation | 2 |
| **2** | **DATA ANALYSIS AND FUNDAMENTAL MACHINE LEARNING TECHNIQUES (UNITS 1 AND 2)** | 4 |
| | 2.1 Overview of Allotted Topics | 4 |
| | 2.2 Dataset Description | 4 |
| | 2.3 Data Preprocessing | 6 |
| | 2.4 Exploratory Data Analysis (EDA) | 8 |
| | 2.5 Proposed System Architecture / Workflow | 12 |
| | 2.6 Algorithm Theory and Mathematical Formulation | 13 |
| | 2.7 Results and Performance Evaluation | 15 |
| | 2.8 Discussion and Findings | 22 |
| | 2.9 Chapter Conclusion | 22 |
| **3** | **UNSUPERVISED LEARNING AND SEQUENTIAL DATA MODELLING (UNITS 3 AND 4)** | 23 |
| | 3.1 Overview of Allotted Unit 3 and Unit 4 Topics | 23 |
| | 3.2 Unit 3 – Dataset Description and Preprocessing | 23 |
| | 3.3 Unit 3 – Theory, EDA, Implementation and Results | 24 |
| | 3.4 Unit 3 – Algorithm Theory and Mathematical Formulation | 25 |
| | 3.5 Unit 3 – Proposed System Architecture / Workflow | 27 |
| | 3.6 Unit 3 – Results and Performance Evaluation | 27 |
| | 3.7 Unit 4 – Sequential Dataset Description | 34 |
| | 3.8 Unit 4 – Sequential Preprocessing and EDA | 35 |
| | 3.9 Unit 4 – Mathematical Formulation | 36 |
| | 3.10 Unit 4 – Proposed System Architecture / Workflow | 38 |
| | 3.11 Results and Performance Evaluation | 38 |
| | 3.12 Discussion, Findings and Limitations | 43 |
| | 3.13 Chapter Conclusion | 43 |
| **4** | **ADVANCED PREDICTIVE MODELLING AND ENSEMBLE LEARNING (UNIT 5)** | 44 |
| | 4.1 Overview of Allotted Unit 5 Topics | 44 |
| | 4.2 Dataset Description and Preprocessing | 44 |
| | 4.3 EDA and Feature Analysis | 45 |
| | 4.4 Proposed Workflow / Architecture | 46 |
| | 4.5 Algorithm Theory and Mathematical Formulation | 47 |
| | 4.6 Results and Performance Evaluation | 49 |
| | 4.7 Comparison with Previous Models | 51 |
| | 4.8 Discussion, Findings and Limitations | 52 |
| | 4.9 Chapter Conclusion | 53 |
| **5** | **INTEGRATED DISCUSSION AND FINDINGS** | 54 |
| **6** | **OVERALL CONCLUSION AND FUTURE SCOPE** | 56 |
| | **REFERENCES** | 58 |
| | **APPENDIX A: CODING** | 60 |

---

## LIST OF FIGURES

| Chapter | Figure | Caption |
|---|---|---|
| 2 | Figure 2.1 | Class balance (left) and road age by outcome (right), Unit 1 |
| 2 | Figure 2.2 | Distributions of rainfall, condition score and traffic load index, split by outcome, Unit 1 |
| 2 | Figure 2.3 | Correlation heatmap of the features and the target, Unit 1 |
| 2 | Figure 2.4 | Advertisement click distribution, Unit 2 |
| 2 | Figure 2.5 | Daily time spent on site versus ad click, Unit 2 |
| 2 | Figure 2.6 | Correlation heatmap, Unit 2 |
| 2 | Figure 2.7 | Daily time spent versus daily internet usage, coloured by click status, Unit 2 |
| 2 | Figure 2.8 | Workflow of the Unit 1 pothole risk prediction system |
| 2 | Figure 2.9 | Workflow of the Unit 2 advertisement click prediction system |
| 2 | Figure 2.10 | Test-set metrics for all four models, Unit 1 |
| 2 | Figure 2.11 | ROC curves of the four models on the test set, Unit 1 |
| 2 | Figure 2.12 | Confusion matrices of the four models at threshold 0.5, Unit 1 |
| 2 | Figure 2.13 | Logistic Regression standardised coefficients (red raises risk, green lowers it), Unit 1 |
| 2 | Figure 2.14 | XGBoost feature importance, Unit 1 |
| 2 | Figure 2.15 | Distribution of predicted probabilities for Logistic Regression with risk cut-offs at 0.33 and 0.66, Unit 1 |
| 2 | Figure 2.16 | Confusion matrix of the Unit 2 Logistic Regression model on the test set |
| 2 | Figure 2.17 | ROC curve of the Unit 2 Logistic Regression model (AUC = 0.98) |
| 2 | Figure 2.18 | Logistic Regression feature coefficients, Unit 2 |
| 3 | Figure 3.1 | The first six Kodak images (kodim01 to kodim06), Unit 3 |
| 3 | Figure 3.2 | kodim01 quantized with K = 4, 8, 16, 32 and 64 (top) and the corresponding palette swatches (bottom) |
| 3 | Figure 3.3 | Workflow of the Unit 3 K-Means colour quantization system |
| 3 | Figure 3.4 | Mean PSNR, mean SSIM and mean compression ratio versus K across 24 images, with error bars |
| 3 | Figure 3.5 | Rate–distortion curve: mean PSNR versus bits per pixel |
| 3 | Figure 3.6 | Elbow curve: inertia versus K for the first three images |
| 3 | Figure 3.7 | Mean silhouette score versus K (first three images, 10,000-pixel samples) |
| 3 | Figure 3.8 | PSNR and SSIM of the four methods at K = 16 |
| 3 | Figure 3.9 | PSNR and SSIM of the four methods at K = 64 |
| 3 | Figure 3.10 | Visual comparison on kodim01: original, K-Means RGB, MiniBatchKMeans, K-Means CIELAB and Median-Cut |
| 3 | Figure 3.11 | Canny edge maps of the original image and of the quantized images for K = 8, 16 and 32 (kodim01) |
| 3 | Figure 3.12 | Best cases at K = 16: originals (left) and quantized images (right) for kodim16, kodim01 and kodim02 |
| 3 | Figure 3.13 | Worst cases at K = 16: originals (left) and quantized images (right) for kodim23, kodim05 and kodim14 |
| 3 | Figure 3.14 | Dataset split distribution for Experiment 2 |
| 3 | Figure 3.15 | Waveform of a “marvin” test recording |
| 3 | Figure 3.16 | MFCC feature map of the same recording (13 coefficients over about 100 frames) |
| 3 | Figure 3.17 | Workflow of the Unit 4 HMM wake-word detection system |
| 3 | Figure 3.18 | Confusion matrix of the HMM wake-word detector, Experiment 1 |
| 3 | Figure 3.19 | Keyword versus background score distribution, Experiment 1 |
| 3 | Figure 3.20 | Wake threshold trade-off between false-accept and false-reject rates on the validation set, Experiment 1 |
| 3 | Figure 3.21 | Learned transition matrices of the keyword HMM and the background HMM, Experiment 2 |
| 3 | Figure 3.22 | ROC curve of the HMM wake-word detector, Experiment 2 (AUC = 0.774) |
| 3 | Figure 3.23 | Wake score distribution by class, Experiment 2 |
| 4 | Figure 4.1 | Workflow of the Unit 5 Bayesian Model Averaging and battery management system |
| 4 | Figure 4.2 | 2019 held-out test performance: MAE of the main models for wind and solar |
| 4 | Figure 4.3 | Battery-management strategies: total cost in the 2019 test year |

---

## LIST OF TABLES

| Chapter | Table | Caption |
|---|---|---|
| 1 | Table 1.1 | Unit-wise topics and datasets |
| 2 | Table 2.1 | Columns of the synthetic pothole dataset |
| 2 | Table 2.2 | Fields of the Advertising dataset |
| 2 | Table 2.3 | Selected summary statistics after cleaning |
| 2 | Table 2.4 | Engineered features for Unit 1 |
| 2 | Table 2.5 | Candidate models and settings, Unit 1 |
| 2 | Table 2.6 | Test-set performance of the four models at threshold 0.5, Unit 1 |
| 2 | Table 2.7 | Five-fold cross-validated ROC-AUC, Unit 1 |
| 2 | Table 2.8 | Risk levels and recommended actions, Unit 1 |
| 2 | Table 2.9 | Actual pothole rate in each risk band on the test set, Unit 1 |
| 2 | Table 2.10 | Effect of the decision threshold for Logistic Regression, Unit 1 |
| 2 | Table 2.11 | Learned coefficients of the standardised model, Unit 2 |
| 2 | Table 2.12 | Summary of Unit 2 results |
| 3 | Table 3.1 | Experimental settings of Unit 3 |
| 3 | Table 3.2 | Mean results over 24 images for four quantizers |
| 3 | Table 3.3 | Edge preservation F1 on kodim01 |
| 3 | Table 3.4 | Best and worst images at K = 16 ranked by PSNR |
| 3 | Table 3.5 | File sizes for kodim01 |
| 3 | Table 3.6 | Selected recording counts per class in the training archive |
| 3 | Table 3.7 | Summary of Unit 3 results |
| 3 | Table 3.8 | Experiment 1 test results at the two thresholds |
| 3 | Table 3.9 | Comparison of the two Unit 4 experiments |
| 4 | Table 4.1 | Dataset elements of Unit 5 |
| 4 | Table 4.2 | Forecast ensemble members trained separately for wind and solar |
| 4 | Table 4.3 | Battery simulation settings and quantile parameters |
| 4 | Table 4.4 | Held-out 2019 forecasting results |
| 4 | Table 4.5 | Combined total-supply uncertainty results |
| 4 | Table 4.6 | Battery simulation results for the 2019 test year |
| 4 | Table 4.7 | Relative improvement of BMA on the 2019 test year |
| 5 | Table 5.1 | Summary of the five units |
| A | Table A.1 | Google Colab notebooks |

---

## CHAPTER 1: INTRODUCTION

### 1.1 Background, Motivation and Scope
Machine learning allows computers to learn patterns from data and to use them for prediction, grouping and decision support. Its value depends on matching the technique to the structure of the data: tabular records call for supervised classifiers, unlabelled image pixels call for clustering, ordered signals such as speech call for sequence models, and forecasts from several sources call for principled ensembles. This report brings together five such studies carried out for the course **21CSC305P Machine Learning Project**, each focusing on a different application domain and a different family of algorithms.

The application domains are road maintenance, online advertising, digital image compression, voice-assistant wake-word detection and renewable-energy forecasting. The main stakeholders are, respectively:
- Road authorities who must prioritise limited maintenance budgets.
- Advertisers who wish to reduce wasted impressions and target engaged users.
- Engineers who store and transmit high-resolution imagery over bandwidth-limited channels.
- Developers of voice assistants who must balance false wake-ups against missed commands.
- Transmission system operators (TSOs) and smart grid managers who must plan day-ahead dispatch around variable wind and solar generation.

The scope of the project is focused on disciplined offline experimentation. Every unit follows the same pattern: loading raw data, performing rigorous preprocessing and exploratory analysis, training models on training partitions only, tuning thresholds/hyperparameters on validation sets, and reporting held-out test metrics.

### 1.2 Integrated Problem Statement
The overall problem addressed by this project is to design, implement and evaluate a set of complementary machine learning techniques on realistic datasets, covering supervised classification, unsupervised clustering, sequential probabilistic modelling and Bayesian ensemble learning, and to report their performance with metrics that are appropriate to each task. The five units were carried out on five different datasets:

1. **Unit 1 (Classification):** Predict, for each road section, whether a pothole will develop within the next 90 days, and convert predicted probabilities into Low, Medium, and High risk tiers for preventive maintenance.
2. **Unit 2 (Classification):** Predict whether a website visitor will click an advertisement using demographic and browsing-behaviour features.
3. **Unit 3 (Unsupervised Learning):** Quantize natural true-colour images into compact palettes of representative colours using K-Means clustering, evaluating the rate–distortion trade-off between bits per pixel and image quality (PSNR, SSIM, edge preservation).
4. **Unit 4 (Sequential Modelling):** Detect the wake word *"marvin"* in 1-second audio clips using Mel-Frequency Cepstral Coefficients (MFCCs) and two continuous Gaussian Hidden Markov Models (keyword vs. background), evaluating the trade-off between false-accept rate (FAR) and false-reject rate (FRR).
5. **Unit 5 (Ensemble Learning & Optimization):** Combine heterogeneous wind and solar forecasts using rolling Bayesian Model Averaging (BMA), model joint uncertainty with a Gaussian copula, and guide state-of-charge adaptive battery commitment policies.

### 1.3 Project Objectives
- **Objective 1 (Unit 1):** Build and benchmark Logistic Regression, Decision Tree, Random Forest, and XGBoost classifiers for 90-day pothole risk prediction.
- **Objective 2 (Unit 2):** Fit an interpretable standardized Logistic Regression model for online ad clicks and assess performance via accuracy, confusion matrix, and ROC-AUC.
- **Objective 3 (Unit 3):** Implement K-Means colour quantization on the Kodak image suite, determine optimal cluster counts $K$, and compare against MiniBatchKMeans, CIELAB K-Means, and Median-Cut.
- **Objective 4 (Unit 4):** Develop a speech keyword spotter using MFCC features and Gaussian HMMs with a custom numerically stable Forward algorithm verified against hmmlearn.
- **Objective 5 (Unit 5):** Formulate a rolling BMA predictive pipeline on German hourly renewable generation from OPSD, derive predictive distributions, and simulate battery dispatch cost savings.

### 1.4 Unit-wise Topic and Dataset Allocation

#### Table 1.1: Unit-wise topics and datasets
| PBL Stage | Unit | Allotted Topic / Algorithm | Dataset | Links |
|---|---|---|---|---|
| **PBL 1** | **Unit 1** | Predictive pothole risk detection: Logistic Regression, Decision Tree, Random Forest, XGBoost | Synthetic road-section dataset (3,000 sections) | [Colab](https://colab.research.google.com/drive/1vdnxaDtFWs3nS3mXF-cxtu_Zd0WCeiUl?usp=sharing) \| [Zenodo Dataset](https://zenodo.org/records/15638682) \| [PPT](https://docs.google.com/presentation/d/1Iu8aryOgr1Jc8gYrUsUqy-s3YyqagVlh/edit?usp=drivesdk&ouid=103491267144499532443&rtpof=true&sd=true) \| [Doc Report](https://docs.google.com/document/d/1IIVtdY49cWUrw07Oku0g4jNRtEbe4PXu/edit?usp=drivesdk&ouid=103491267144499532443&rtpof=true&sd=true) |
| **PBL 1** | **Unit 2** | Advertisement click prediction: Logistic Regression | Advertising dataset, Kaggle (1,000 rows, 10 columns) | [Colab](https://colab.research.google.com/drive/1ZQLfjucPS5IzBeQduiwBqBOjK8kLYam5?usp=sharing) \| [Kaggle Dataset](https://www.kaggle.com/datasets/abhinavgautam/advertising/data) \| [PPT](https://docs.google.com/presentation/d/1qyjvYdyOY2Se6DihbYU0HQPRom9LxRbw/edit?usp=drive_link&ouid=111278213256287944154&rtpof=true&sd=true) \| [Doc Report](https://drive.google.com/file/d/1dk-MmbkNZNwkGjCrc4AJPMW54zJ6Yw18/view?usp=sharing) |
| **PBL 2** | **Unit 3** | Colour quantization: K-Means clustering | Kodak image set (24 natural images) | [Colab](https://colab.research.google.com/drive/1peFHJ_pltOn2WFwBpjjo5KmXxWeiS_YC) \| [Kodak Dataset](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset?resource=download) \| [PPT](https://docs.google.com/presentation/d/1b4KSjXLuSFu7JxQbJvyux4Fx_ZqieSB-/edit?slide=id.p1#slide=id.p1) \| [Doc Report](https://docs.google.com/document/d/1cTH5HUS1s7sarJOzHA2ERPv2vF0INB4f/edit?usp=sharing&ouid=116814814621085145629&rtpof=true&sd=true) |
| **PBL 2** | **Unit 4** | Wake-word detection: Gaussian Hidden Markov Models, Forward algorithm | Google Speech Commands dataset (WAV audio) | [Colab](https://colab.research.google.com/drive/1bEm9wQAYaoB6NiM4J8aKbIFr-o0_oClH?usp=sharing) \| [Kaggle Speech Dataset](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data) \| [PPT](https://docs.google.com/presentation/d/1ddGflHi6onEtdWRFuP1YxXTWThx1g_rM/edit?usp=sharing&ouid=104414554011739467437&rtpof=true&sd=true) \| [Doc Report](https://docs.google.com/document/d/1E56p1b5ISfX79VZ75IS2kaL1_e5f5iia/edit?usp=sharing&ouid=104414554011739467437&rtpof=true&sd=true) |
| **PBL 3** | **Unit 5** | Renewable forecasting and battery management: Bayesian Model Averaging | Open Power System Data (OPSD) Germany (2015-2019) | [Colab](https://colab.research.google.com/drive/1rxozy93DZf-Te1kDaeszMibY_P9cVio0?usp=sharing) \| [OPSD Time Series](https://data.open-power-system-data.org/time_series/) \| [PPT](https://docs.google.com/presentation/d/1q3EaaWO-0cwhWasm_YJNTrxsoYcu13Df/edit?usp=sharing&ouid=103491267144499532443&rtpof=true&sd=true) \| [Doc Report](https://drive.google.com/file/d/1L4nI8YPlvPMmnPAP1009QYJh5OC3R_ua/view?usp=sharing) |

---

## CHAPTER 2: DATA ANALYSIS AND FUNDAMENTAL MACHINE LEARNING TECHNIQUES (UNITS 1 AND 2)

### 2.1 Overview of Allotted Topics
PBL 1 investigates supervised binary classification on structured tabular records. Unit 1 explores predictive infrastructure maintenance, while Unit 2 addresses visitor conversion prediction in digital advertising.

### 2.2 Dataset Description

#### 2.2.1 Unit 1 – Synthetic Road-Section Dataset
A synthetic dataset of 3,000 road sections was constructed with 13 predictor columns, a section identifier, and a binary target `pothole_next_90d`.

##### Table 2.1: Columns of the synthetic pothole dataset
| Category | Column | Description |
|---|---|---|
| Weather | `rainfall_mm_90d` | Accumulated rainfall over prior 90 days (mm) |
| Weather | `temp_variation_c` | Diurnal temperature swing (°C) |
| Traffic | `traffic_volume` | Vehicles per day |
| Traffic | `heavy_vehicle_pct` | Share of heavy commercial vehicles (%) |
| Road attributes | `road_age` | Years since construction |
| Road attributes | `surface_material` | Asphalt, Concrete, or Chip-seal |
| Road attributes | `segment_length_km` | Length of monitored section (km) |
| Condition | `condition_score` | Surface rating from 0 to 100 (higher = better) |
| Condition | `crack_density` | Fraction of cracked surface area |
| Condition | `condition_change` | Change rate of condition score per period |
| Maintenance | `days_since_maint` | Elapsed days since last structural repair |
| Maintenance | `maint_per_year` | Historical maintenance frequency per year |
| Pothole history | `prior_potholes` | Count of previously recorded potholes |
| Target | `pothole_next_90d` | 1 if pothole develops within 90 days, else 0 |

The ground truth target was generated via:
$$p(x) = \frac{1}{1 + e^{-\eta}}, \quad y \sim \text{Bernoulli}(p(x))$$
$$\eta = \beta_0 + \sum \beta_j z_j + \beta_{rt} z_{\text{rain}} z_{\text{traffic}} + \beta_{ac} z_{\text{age}} (-z_{\text{condition}})$$

#### 2.2.2 Unit 2 – Advertising Dataset
The Kaggle Advertising dataset comprises 1,000 observations across 10 fields with balanced classes (500 clicked, 500 did not).

##### Table 2.2: Fields of the Advertising dataset
| Type | Fields | Role |
|---|---|---|
| Predictors | `Age`, `Area Income`, `Daily Time Spent on Site`, `Daily Internet Usage`, `Male` | Model inputs |
| Target | `Clicked on Ad` | Binary outcome (0 = No Click, 1 = Click) |
| Excluded | `Ad Topic Line`, `City`, `Country`, `Timestamp` | Metadata / Unstructured |

### 2.3 Data Preprocessing

#### 2.3.1 Unit 1 Preprocessing & Engineering
Missing values were imputed with column medians. Extreme traffic outliers exceeding $Q_3 + 3 \times \text{IQR}$ were clipped.

##### Table 2.4: Engineered features for Unit 1
| Feature | Formula | Physical Rationale |
|---|---|---|
| `traffic_load_index` | $\text{volume} \times (1 + 2 \times \text{heavy\%} / 100)$ | Heavy axles exert disproportionate pavement stress |
| `maint_delay_ratio` | $\text{days\_since\_maint} / 365$ | Unaddressed wear in years |
| `wetness_stress` | $\text{rainfall} \times \text{temp\_variation} / 100$ | Freeze-thaw pore pressure cycle |
| `mat_Concrete`, `mat_Chip-seal` | One-hot encoded (Asphalt baseline) | Differential structural resilience |

### 2.4 Exploratory Data Analysis (EDA)

#### Unit 1 Visualizations
<div align="center">
  <img src="figures/fig_2_1_target_age.png" alt="Class Balance and Road Age" width="600"/>
  <p><em>Figure 2.1: Class balance (left) and road age by outcome (right), Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_2_distributions.png" alt="Feature Distributions" width="650"/>
  <p><em>Figure 2.2: Distributions of rainfall, condition score and traffic load index, split by outcome, Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_3_correlation_matrix.png" alt="Unit 1 Correlation Heatmap" width="650"/>
  <p><em>Figure 2.3: Correlation heatmap of the features and the target, Unit 1</em></p>
</div>

#### Unit 2 Visualizations
<div align="center">
  <img src="figures/fig_2_4_click_distribution.png" alt="Ad Click Distribution" width="450"/>
  <p><em>Figure 2.4: Advertisement click distribution, Unit 2</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_5_time_vs_click.png" alt="Time Spent vs Click" width="500"/>
  <p><em>Figure 2.5: Daily time spent on site versus ad click, Unit 2</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_6_ad_correlation.png" alt="Unit 2 Correlation Heatmap" width="550"/>
  <p><em>Figure 2.6: Correlation heatmap, Unit 2</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_7_time_vs_internet.png" alt="Time Spent vs Internet Usage" width="600"/>
  <p><em>Figure 2.7: Daily time spent versus daily internet usage, coloured by click status, Unit 2</em></p>
</div>

### 2.5 System Workflows
<div align="center">
  <img src="figures/fig_2_8_pothole_workflow.png" alt="Unit 1 Workflow" width="550"/>
  <p><em>Figure 2.8: Workflow of the Unit 1 pothole risk prediction system</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_9_ad_workflow.png" alt="Unit 2 Workflow" width="550"/>
  <p><em>Figure 2.9: Workflow of the Unit 2 advertisement click prediction system</em></p>
</div>

### 2.6 Algorithm Theory & Mathematical Formulation
#### Logistic Regression
$$\log\left(\frac{p}{1-p}\right) = w_0 + \sum_{j=1}^d w_j x_j, \qquad p = \frac{1}{1 + e^{-z}}$$
$$\mathcal{L}(w) = -\frac{1}{N}\sum_{i=1}^N \left[ y_i \log p_i + (1 - y_i)\log(1 - p_i) \right]$$

### 2.7 Results and Performance Evaluation

#### Unit 1 Results
##### Table 2.6: Test-set performance of the four models at threshold 0.5, Unit 1
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.737** | **0.605** | **0.714** | **0.655** | **0.806** |
| Decision Tree | 0.655 | 0.507 | 0.490 | 0.499 | 0.683 |
| Random Forest | 0.740 | 0.638 | 0.595 | 0.616 | 0.795 |
| XGBoost | 0.758 | 0.721 | 0.505 | 0.594 | 0.801 |

<div align="center">
  <img src="figures/fig_2_10_model_comparison.png" alt="Model Comparison" width="650"/>
  <p><em>Figure 2.10: Test-set metrics for all four models, Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_11_pothole_roc.png" alt="ROC Curves Unit 1" width="500"/>
  <p><em>Figure 2.11: ROC curves of the four models on the test set, Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_12_pothole_confusion.png" alt="Confusion Matrices Unit 1" width="700"/>
  <p><em>Figure 2.12: Confusion matrices of the four models at threshold 0.5, Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_13_lr_coefficients.png" alt="LR Coefficients" width="500"/>
  <p><em>Figure 2.13: Logistic Regression standardised coefficients, Unit 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_14_xgb_importance.png" alt="XGBoost Feature Importance" width="500"/>
  <p><em>Figure 2.14: XGBoost feature importance, Unit 1</em></p>
</div>

##### Table 2.8: Risk levels and recommended actions, Unit 1
| Risk Level | Probability Range | Recommended Action |
|---|---|---|
| **Low** | 0.00 to 0.33 | Routine sensor monitoring |
| **Medium** | 0.33 to 0.66 | Schedule maintenance inspection within 30 days |
| **High** | 0.66 to 1.00 | Immediate preventive resurfacing and drainage clearance |

##### Table 2.9: Actual pothole rate in each risk band on the test set, Unit 1
| Risk Band | Road Sections | Share that Developed a Pothole |
|---|---|---|
| **Low** | 225 | 12.0% |
| **Medium** | 242 | 37.2% |
| **High** | 133 | 69.9% |

<div align="center">
  <img src="figures/fig_2_15_probability_distribution.png" alt="Probability Distribution" width="500"/>
  <p><em>Figure 2.15: Distribution of predicted probabilities for Logistic Regression with risk cut-offs at 0.33 and 0.66, Unit 1</em></p>
</div>

#### Unit 2 Results
<div align="center">
  <img src="figures/fig_2_16_ad_confusion.png" alt="Unit 2 Confusion Matrix" width="450"/>
  <p><em>Figure 2.16: Confusion matrix of the Unit 2 Logistic Regression model on the test set</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_17_ad_roc.png" alt="Unit 2 ROC Curve" width="450"/>
  <p><em>Figure 2.17: ROC curve of the Unit 2 Logistic Regression model (AUC = 0.98)</em></p>
</div>

<div align="center">
  <img src="figures/fig_2_18_ad_coefficients.png" alt="Unit 2 Feature Coefficients" width="550"/>
  <p><em>Figure 2.18: Logistic Regression feature coefficients, Unit 2</em></p>
</div>

##### Table 2.11: Learned coefficients of the standardised model, Unit 2
| Feature | Standardized Coefficient |
|---|---|
| `Age` | +1.2587 |
| `Area Income` | -1.6336 |
| `Daily Time Spent on Site` | -2.5332 |
| `Daily Internet Usage` | -2.6987 |
| `Male` | -0.3587 |

---

## CHAPTER 3: UNSUPERVISED LEARNING AND SEQUENTIAL DATA MODELLING (UNITS 3 AND 4)

### 3.1 Overview
PBL 2 progresses from structured tabular classification to unlabelled image data (clustering) and continuous temporal speech signals (sequence modelling).

### 3.2 Unit 3 – Colour Quantization (K-Means)
The Kodak image suite (24 uncompressed natural images of shape $512 \times 768 \times 3$) was used to benchmark color quantization.

<div align="center">
  <img src="figures/fig_3_1_kodak_samples.jpg" alt="Kodak Image Suite" width="650"/>
  <p><em>Figure 3.1: The first six Kodak images (kodim01 to kodim06), Unit 3</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_2_quantized_swatches.jpg" alt="Quantized Swatches" width="650"/>
  <p><em>Figure 3.2: kodim01 quantized with K = 4, 8, 16, 32 and 64 (top) and the corresponding palette swatches (bottom)</em></p>
</div>

#### Workflow
<div align="center">
  <img src="figures/fig_3_3_kmeans_workflow.png" alt="K-Means Workflow" width="500"/>
  <p><em>Figure 3.3: Workflow of the Unit 3 K-Means colour quantization system</em></p>
</div>

#### Rate-Distortion & Quality Evaluation
$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$
$$\text{PSNR} = 20\log_{10}\left(\frac{255}{\sqrt{\text{MSE}}}\right), \qquad \text{Ratio} = \frac{24 \cdot H \cdot W}{H \cdot W \cdot \lceil \log_2 K \rceil + 24 K}$$

<div align="center">
  <img src="figures/fig_3_4_psnr_ssim_curves.png" alt="PSNR SSIM Ratio vs K" width="700"/>
  <p><em>Figure 3.4: Mean PSNR, mean SSIM and mean compression ratio versus K across 24 images, with error bars</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_5_rate_distortion.png" alt="Rate Distortion Curve" width="500"/>
  <p><em>Figure 3.5: Rate–distortion curve: mean PSNR versus bits per pixel</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_6_elbow_curve.png" alt="Elbow Curve" width="500"/>
  <p><em>Figure 3.6: Elbow curve: inertia versus K for the first three images</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_7_silhouette_score.png" alt="Silhouette Score" width="500"/>
  <p><em>Figure 3.7: Mean silhouette score versus K (first three images, 10,000-pixel samples)</em></p>
</div>

##### Table 3.2: Mean results over 24 images for four quantizers
| K | Method | PSNR (dB) | SSIM | Ratio | Time (s) |
|---|---|---|---|---|---|
| **16** | **K-Means RGB** | **30.21** | **0.881** | **6.00** | **0.238** |
| 16 | MiniBatchKMeans RGB | 30.01 | 0.879 | 6.00 | 0.062 |
| 16 | K-Means CIELAB | 28.68 | 0.836 | 6.00 | 0.358 |
| 16 | Median-Cut | 27.18 | 0.876 | 6.00 | 0.096 |
| **64** | **K-Means RGB** | **35.86** | **0.951** | **4.00** | **0.877** |
| 64 | MiniBatchKMeans RGB | 35.59 | 0.950 | 4.00 | 0.169 |
| 64 | K-Means CIELAB | 33.82 | 0.920 | 4.00 | 1.042 |
| 64 | Median-Cut | 32.07 | 0.941 | 4.00 | 0.097 |

<div align="center">
  <img src="figures/fig_3_8_methods_k16.png" alt="Methods K=16" width="600"/>
  <p><em>Figure 3.8: PSNR and SSIM of the four methods at K = 16</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_9_methods_k64.png" alt="Methods K=64" width="600"/>
  <p><em>Figure 3.9: PSNR and SSIM of the four methods at K = 64</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_10_visual_comparison.jpg" alt="Visual Comparison" width="700"/>
  <p><em>Figure 3.10: Visual comparison on kodim01: original, K-Means RGB, MiniBatchKMeans, K-Means CIELAB and Median-Cut</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_11_canny_edges.jpg" alt="Canny Edges" width="700"/>
  <p><em>Figure 3.11: Canny edge maps of the original image and of the quantized images for K = 8, 16 and 32 (kodim01)</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_12_best_cases.jpg" alt="Best Cases K=16" width="600"/>
  <p><em>Figure 3.12: Best cases at K = 16: originals (left) and quantized images (right) for kodim16, kodim01 and kodim02</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_13_worst_cases.jpg" alt="Worst Cases K=16" width="600"/>
  <p><em>Figure 3.13: Worst cases at K = 16: originals (left) and quantized images (right) for kodim23, kodim05 and kodim14</em></p>
</div>

---

### 3.3 Unit 4 – Sequential Wake-Word Detection (HMMs)
The Google Speech Commands dataset was used to spot the wake-word *"marvin"* against background speech.

<div align="center">
  <img src="figures/fig_3_14_speech_split.png" alt="Dataset Split" width="500"/>
  <p><em>Figure 3.14: Dataset split distribution for Experiment 2</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_15_waveform.png" alt="Audio Waveform" width="650"/>
  <p><em>Figure 3.15: Waveform of a “marvin” test recording</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_16_mfcc_map.png" alt="MFCC Feature Map" width="650"/>
  <p><em>Figure 3.16: MFCC feature map of the same recording (13 coefficients over about 100 frames)</em></p>
</div>

#### System Architecture & Forward Algorithm
<div align="center">
  <img src="figures/fig_3_17_hmm_workflow.png" alt="HMM Workflow" width="500"/>
  <p><em>Figure 3.17: Workflow of the Unit 4 HMM wake-word detection system</em></p>
</div>

$$\alpha_1(j) = \pi_j \cdot b_j(o_1), \qquad \alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right] b_j(o_t)$$
$$S(X) = \frac{1}{T}\left[ \log P(X \mid \lambda_{\text{keyword}}) - \log P(X \mid \lambda_{\text{background}}) \right]$$

<div align="center">
  <img src="figures/fig_3_18_hmm_confusion_exp1.png" alt="Confusion Matrix Exp 1" width="450"/>
  <p><em>Figure 3.18: Confusion matrix of the HMM wake-word detector, Experiment 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_19_score_dist_exp1.png" alt="Score Distribution Exp 1" width="600"/>
  <p><em>Figure 3.19: Keyword versus background score distribution, Experiment 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_20_far_frr_tradeoff.png" alt="Threshold Tradeoff" width="550"/>
  <p><em>Figure 3.20: Wake threshold trade-off between false-accept and false-reject rates on the validation set, Experiment 1</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_21_transition_matrices.png" alt="Transition Matrices" width="600"/>
  <p><em>Figure 3.21: Learned transition matrices of the keyword HMM and the background HMM, Experiment 2</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_22_hmm_roc_exp2.png" alt="ROC Exp 2" width="450"/>
  <p><em>Figure 3.22: ROC curve of the HMM wake-word detector, Experiment 2 (AUC = 0.774)</em></p>
</div>

<div align="center">
  <img src="figures/fig_3_23_score_boxplots.png" alt="Boxplot Exp 2" width="500"/>
  <p><em>Figure 3.23: Wake score distribution by class, Experiment 2</em></p>
</div>

##### Table 3.9: Comparison of the two Unit 4 experiments
| Metric | Experiment 1 (Balanced) | Experiment 2 (Expanded Background) | Change |
|---|---|---|---|
| Test Set Size | 300 | 600 | - |
| Accuracy | 70.0% | 72.8% | +2.8 pts |
| Keyword Recall | 88.7% (133/150) | 60.7% (91/150) | -28.0 pts |
| Keyword Precision | 0.65 | 0.47 | -0.18 |
| False-Accept Rate (FAR) | 48.7% (73/150) | 23.1% (104/450) | -25.6 pts |
| Validation F1 | 0.777 | 0.523 | -0.254 |

---

## CHAPTER 4: ADVANCED PREDICTIVE MODELLING AND ENSEMBLE LEARNING (UNIT 5)

### 4.1 Overview
Unit 5 investigates Bayesian Model Averaging (BMA) for combining heterogeneous day-ahead forecasts of German wind and solar generation from the Open Power System Data (OPSD) platform (2015-2019).

<div align="center">
  <img src="figures/fig_4_1_bma_workflow.png" alt="BMA Workflow" width="550"/>
  <p><em>Figure 4.1: Workflow of the Unit 5 Bayesian Model Averaging and battery management system</em></p>
</div>

### 4.2 Algorithm Theory
$$p(y \mid f_1, \dots, f_K) = \sum_{k=1}^K w_k \cdot \mathcal{N}\left(a_k + b_k f_k, \sigma^2\right)$$
$$\tau^* = \frac{c_{\text{spill}}}{c_{\text{spill}} + c_{\text{short}}} = \frac{30}{30 + 120} = 0.20$$

### 4.3 Results and Performance Evaluation
<div align="center">
  <img src="figures/fig_4_2_forecast_mae.png" alt="Forecast MAE" width="600"/>
  <p><em>Figure 4.2: 2019 held-out test performance: MAE of the main models for wind and solar</em></p>
</div>

##### Table 4.4: Held-out 2019 forecasting results (capacity-factor units)
| Target | Model | MAE | RMSE | CRPS | 90% Cover | 90% Width |
|---|---|---|---|---|---|---|
| **Wind** | Persistence_24h | 0.1387 | 0.1823 | – | – | – |
| Wind | Mean_7d | 0.1443 | 0.1859 | – | – | – |
| Wind | Ridge | 0.0678 | 0.0850 | – | – | – |
| Wind | RandomForest | 0.0683 | 0.0883 | – | – | – |
| Wind | GradBoost | 0.0683 | 0.0879 | – | – | – |
| Wind | GradBoost_weather_only | 0.0686 | 0.0886 | – | – | – |
| Wind | **BMA** | **0.0617** | **0.0783** | **0.0439** | **0.8800** | **0.2401** |
| Wind | Equal-weight avg | 0.0759 | 0.0927 | 0.0529 | 0.9763 | 0.4110 |
| **Solar** | Persistence_24h | 0.0405 | 0.0665 | – | – | – |
| Solar | Mean_7d | 0.0444 | 0.0677 | – | – | – |
| Solar | Ridge | 0.0288 | 0.0394 | – | – | – |
| Solar | RandomForest | 0.0253 | 0.0388 | – | – | – |
| Solar | GradBoost | 0.0259 | 0.0397 | – | – | – |
| Solar | GradBoost_weather_only | 0.0272 | 0.0406 | – | – | – |
| Solar | **BMA** | **0.0211** | **0.0315** | **0.0161** | **0.8853** | **0.0881** |
| Solar | Equal-weight avg | 0.0243 | 0.0361 | 0.0196 | 0.9721 | 0.1577 |

<div align="center">
  <img src="figures/fig_4_3_battery_cost.png" alt="Battery Cost Comparison" width="600"/>
  <p><em>Figure 4.3: Battery-management strategies: total cost in the 2019 test year</em></p>
</div>

##### Table 4.6: Battery simulation results for the 2019 test year
| Strategy | Cost (kEUR) | Shortage (MWh) | Spilled (MWh) | Cycles | Saving vs Best Single |
|---|---|---|---|---|---|
| BMA mean (battery) | 1641.6 | 10683.0 | 9845.6 | 80.3 | 38.9% |
| Equal-weight mean (battery) | 2296.5 | 15167.0 | 14039.0 | 69.2 | 14.5% |
| Best single model (RF/RF), battery | 2687.3 | 20753.3 | 4424.3 | 80.2 | 0.0% |
| BMA quantile, no battery | 1800.9 | 7433.3 | 30295.5 | 0.0 | 33.0% |
| BMA quantile $\tau = 0.28$, battery | 1107.4 | 2568.0 | 24938.3 | 63.9 | 58.8% |
| **Adaptive BMA $\tau_0 = 0.26, \kappa = 0.6$, battery** | **504.3** | **249.1** | **13461.8** | **88.2** | **81.2%** |

---

## CHAPTER 5: INTEGRATED DISCUSSION AND FINDINGS

##### Table 5.1: Summary of the five units
| Unit | Task and Technique | Dataset | Headline Quantitative Result |
|---|---|---|---|
| **1** | 90-day pothole risk; LR, DT, RF, XGBoost | Synthetic, 3,000 sections | LR best: ROC-AUC 0.806, Recall 0.714; High-risk band has 69.9% pothole rate |
| **2** | Ad click prediction; Logistic Regression | Advertising, 1,000 rows | Accuracy 96%, ROC-AUC 0.98; Daily internet usage ($-2.70$) strongest driver |
| **3** | Colour quantization; K-Means | Kodak, 24 images | $K = 16$: Ratio 6.0, PSNR 30.21 dB, SSIM 0.881; 85% storage reduction |
| **4** | Wake-word detection; Gaussian HMMs | Speech Commands | Accuracy 72.8%, ROC-AUC 0.774, FAR 23.1%, Recall 60.7% |
| **5** | Forecasting and battery; BMA | OPSD Germany, 2015-2019 | Wind MAE 0.0617, Solar MAE 0.0211, Battery Cost 504.3 kEUR (81.2% savings) |

---

## CHAPTER 6: OVERALL CONCLUSION AND FUTURE SCOPE

The integrated project successfully formulated, implemented, and validated five foundational machine learning paradigms across real-world application domains:
1. **Unit 1:** Proved that linear models with physical feature engineering match or exceed complex ensembles when deterioration mechanisms exhibit additive properties.
2. **Unit 2:** Established high-accuracy visitor targeting (96% accuracy, 0.98 AUC) using interpretable log-odds coefficients.
3. **Unit 3:** Demonstrated that K-Means colour quantization at $K = 16$ achieves a 6-fold data reduction while maintaining structural fidelity (SSIM = 0.881, PSNR = 30.21 dB).
4. **Unit 4:** Implemented a continuous Gaussian HMM speech spotter with a library-verified Forward algorithm, demonstrating operational threshold trade-offs.
5. **Unit 5:** Demonstrated that rolling Bayesian Model Averaging outperforms individual models and equal-weight averaging, delivering an 81.2% operational cost reduction in battery energy dispatch.

### Future Scope
- **Unit 1:** Integration with municipal LiDAR, computer vision road inspection vehicles, and asphalt temperature sensors.
- **Unit 2:** Text embeddings for ad topic lines, geotargeting encoding, and real-time impression bidding.
- **Unit 3:** Adaptive spatial quadtree quantization, perceptual color spaces (CIEDE2000), and Floyd-Steinberg error diffusion dithering.
- **Unit 4:** Deep recurrent neural networks (CRNN, Conformer), left-to-right HMM topologies, and noise-robust streaming inference.
- **Unit 5:** Physics-informed neural networks (PINNs), Model Predictive Control (MPC) battery dispatch, and multi-market arbitrage.

---

## REFERENCES
1. L. Breiman, “Random forests,” Machine Learning, vol. 45, no. 1, pp. 5–32, 2001.
2. T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. 22nd ACM SIGKDD, 2016, pp. 785–794.
3. F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” JMLR, vol. 12, pp. 2825–2830, 2011.
4. A. Gautam, “Advertising dataset,” Kaggle, 2018.
5. T. Hastie, R. Tibshirani, and J. Friedman, The Elements of Statistical Learning, 2nd ed. Springer, 2009.
6. S. Mehta, “Kodak dataset,” Kaggle, 2019.
7. R. Franzen, “Kodak lossless true color image suite,” 1999.
8. S. P. Lloyd, “Least squares quantization in PCM,” IEEE Trans. Inf. Theory, vol. 28, no. 2, pp. 129–137, 1982.
9. D. Arthur and S. Vassilvitskii, “k-means++: The advantages of careful seeding,” in SODA, 2007, pp. 1027–1035.
10. Z. Wang et al., “Image quality assessment: From error visibility to structural similarity,” IEEE TIP, vol. 13, no. 4, 2004.
11. P. J. Rousseeuw, “Silhouettes: A graphical aid to the interpretation of cluster analysis,” CAM, vol. 20, pp. 53–65, 1987.
12. J. Canny, “A computational approach to edge detection,” IEEE TPAMI, vol. 8, no. 6, pp. 679–698, 1986.
13. D. Sculley, “Web-scale k-means clustering,” in Proc. 19th WWW, 2010, pp. 1177–1178.
14. P. Heckbert, “Color image quantization for frame buffer display,” ACM SIGGRAPH, vol. 16, no. 3, pp. 297–307, 1982.
15. P. Warden, “Speech commands: A dataset for limited-vocabulary speech recognition,” arXiv:1804.03209, 2018.
16. S. Davis and P. Mermelstein, “Comparison of parametric representations for monosyllabic word recognition,” IEEE TASSP, 1980.
17. L. R. Rabiner, “A tutorial on hidden Markov models and selected applications in speech recognition,” Proc. IEEE, 1989.
18. L. E. Baum et al., “A maximization technique occurring in the statistical analysis of probabilistic functions,” Ann. Math. Stat., 1970.
19. Open Power System Data, “Time series package, Germany, hourly data,” 2020.
20. Open-Meteo, “Historical weather API,” 2023.
21. A. E. Raftery et al., “Using Bayesian model averaging to calibrate forecast ensembles,” Monthly Weather Review, 2005.
22. A. P. Dempster et al., “Maximum likelihood from incomplete data via the EM algorithm,” JRSS-B, vol. 39, no. 1, 1977.
23. T. Gneiting and A. E. Raftery, “Strictly proper scoring rules, prediction, and estimation,” JASA, vol. 102, no. 477, 2007.

---

## APPENDIX A: CODING & REPOSITORY ARTIFACTS
Complete executable Python implementations and Jupyter Notebooks are provided within the project repository:
- `Unit_1_Pothole_Risk_Prediction/pothole_prediction.py`
- `Unit_2_Ad_Click_Prediction/ad_click_prediction.py`
- `Unit_3_Colour_Quantization_KMeans/colour_quantization.py`
- `Unit_4_Wake_Word_Detection_HMM/wake_word_hmm.py`
- `Unit_5_BMA_Renewable_Forecasting/bma_renewable_battery.py`
