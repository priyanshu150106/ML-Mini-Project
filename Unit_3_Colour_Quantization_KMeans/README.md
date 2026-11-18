# Unit 3: Image Colour Quantization via K-Means Clustering

## Overview
This unit investigates unsupervised color quantization, framing image data reduction as pixel clustering in three-dimensional color spaces to find an optimal palette of $K$ representative centroids.

## Key Links
- **Google Colab Notebook:** [Colab Implementation](https://colab.research.google.com/drive/1peFHJ_pltOn2WFwBpjjo5KmXxWeiS_YC)
- **Dataset:** [Kodak Lossless True Colour Suite (Kaggle)](https://www.kaggle.com/datasets/sherylmehta/kodak-dataset?resource=download) (24 natural images)
- **Presentation:** [Google Slides](https://docs.google.com/presentation/d/1b4KSjXLuSFu7JxQbJvyux4Fx_ZqieSB-/edit?slide=id.p1#slide=id.p1)
- **Unit Report:** [Google Docs Report](https://docs.google.com/document/d/1cTH5HUS1s7sarJOzHA2ERPv2vF0INB4f/edit?usp=sharing&ouid=116814814621085145629&rtpof=true&sd=true)

## Quantization Algorithms Compared
1. **K-Means (RGB):** Standard Lloyd's algorithm with k-means++ initialization.
2. **MiniBatchKMeans (RGB):** Fast mini-batch centroid updates.
3. **K-Means (CIELAB):** Perceptually uniform color distance optimization.
4. **Median-Cut:** Recursive spatial bounding box division (Pillow baseline).

## Rate-Distortion & Quality Metrics
Across 24 test images ($512 \times 768$ pixels):
- **Selected Operating Point ($K = 16$):**
  - **PSNR:** 30.21 dB (reconstruction fidelity)
  - **SSIM:** 0.881 (structural similarity)
  - **Edge Preservation $F_1$:** 0.894 (Canny edge preservation)
  - **Bits Per Pixel (BPP):** 4.0 bpp
  - **Compression Ratio:** 6.0x theoretical reduction
  - **Physical File Size:** 110.56 KB (indexed PNG) vs. 762.14 KB (raw 24-bit PNG) $\rightarrow$ **85% storage reduction**.

## Comparison at $K = 16$
| Method | PSNR (dB) | SSIM | Ratio | Time per Image (s) |
|---|---|---|---|---|
| **K-Means RGB** | **30.21** | **0.881** | **6.00** | 0.238 |
| MiniBatchKMeans RGB | 30.01 | 0.879 | 6.00 | **0.062** ($3.8\times$ faster) |
| K-Means CIELAB | 28.68 | 0.836 | 6.00 | 0.358 |
| Median-Cut | 27.18 | 0.876 | 6.00 | 0.096 |

## How to Run
```bash
python colour_quantization.py
```
Or open `Unit_3_Colour_Quantization.ipynb` in Jupyter Notebook / Google Colab.
