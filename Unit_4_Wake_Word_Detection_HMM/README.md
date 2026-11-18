# Unit 4: Wake-Word Detection via Gaussian Hidden Markov Models

## Overview
This unit implements an acoustic wake-word spotting system for the trigger word *"marvin"* using continuous Gaussian Hidden Markov Models (HMMs) over temporal sequences of Mel-Frequency Cepstral Coefficients (MFCCs).

## Key Links
- **Google Colab Notebook:** [Colab Implementation](https://colab.research.google.com/drive/1bEm9wQAYaoB6NiM4J8aKbIFr-o0_oClH?usp=sharing)
- **Dataset:** [Google Speech Commands Dataset (Kaggle)](https://www.kaggle.com/competitions/tensorflow-speech-recognition-challenge/data)
- **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1ddGflHi6onEtdWRFuP1YxXTWThx1g_rM/edit?usp=sharing&ouid=104414554011739467437&rtpof=true&sd=true)
- **Unit Report:** [Google Docs Report](https://docs.google.com/document/d/1E56p1b5ISfX79VZ75IS2kaL1_e5f5iia/edit?usp=sharing&ouid=104414554011739467437&rtpof=true&sd=true)

## Architecture & Algorithm
- **Audio Processing:** 16 kHz mono WAV clips $\rightarrow$ 13 MFCC coefficients (25 ms Hamming window, 10 ms hop) $\rightarrow$ $101 \times 13$ temporal frame sequence.
- **Probabilistic Modeling:**
  - Keyword HMM ($\lambda_k$): 4 hidden states with diagonal Gaussian emissions.
  - Background HMM ($\lambda_b$): 4 hidden states covering non-keyword speech.
- **Log-Space Forward Algorithm:**
  $$\alpha_1(j) = \pi_j \cdot b_j(o_1), \qquad \alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right] b_j(o_t)$$
  Hand-written implementation using `scipy.special.logsumexp` verified to match `hmmlearn` with zero deviation ($\Delta = 0.0$).
- **Decision Metric:**
  $$S(X) = \frac{1}{T}\left[ \log P(X \mid \lambda_k) - \log P(X \mid \lambda_b) \right] \ge \tau$$

## Quantitative Performance
| Experiment | Test Size | Class Balance | Accuracy | Keyword Recall | False-Accept Rate (FAR) | ROC-AUC |
|---|---|---|---|---|---|---|
| **Exp 1 (Balanced)** | 300 | 1:1 (150 kw / 150 bg) | 70.0% | 88.7% | 48.7% | 0.770 |
| **Exp 2 (Expanded Background)** | 600 | 1:3 (150 kw / 450 bg) | **72.8%** | **60.7%** | **23.1%** | **0.774** |

## Operating Threshold Trade-Offs (Exp 1)
- **F1-Optimal Threshold ($\tau = -0.136$):** High Keyword Recall (88.7%), but higher FAR (48.7%).
- **Constrained FAR $\le 10\%$ Threshold ($\tau = 0.888$):** Drastic reduction in false alarms ($\text{FAR} = 7.3\%$), at the expense of Keyword Recall (39.3%).

## How to Run
```bash
python wake_word_hmm.py --exp 2
```
Or open `Unit_4_Wake_Word_HMM.ipynb` in Jupyter Notebook / Google Colab.
