"""
Unit 3: Image Colour Quantization via K-Means Clustering
Part of 21CSC305P Machine Learning Mini Project
Academic Year: 2026-2027 | SRM Institute of Science and Technology

Problem Statement:
Reduce the continuous colour space of natural photographic images to a discrete palette
of representative colours using K-Means clustering, and analyze the rate-distortion
trade-off between memory footprint (bits per pixel, compression ratio) and perceptual fidelity
(PSNR, SSIM, edge preservation).

Methods Compared:
1. K-Means (RGB space)
2. MiniBatchKMeans (RGB space, accelerated batch updates)
3. K-Means (CIELAB perceptual color space)
4. Median-Cut (Pillow baseline)
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans, MiniBatchKMeans
import skimage.metrics
from skimage import color, feature

def get_sample_images(image_dir="data/kodak", num_images=3):
    os.makedirs(image_dir, exist_ok=True)
    existing = [
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    if existing:
        return sorted(existing)[:num_images]

    # Try downloading from kagglehub
    try:
        import kagglehub
        print("Downloading Kodak dataset via kagglehub...")
        path = kagglehub.dataset_download("sherylmehta/kodak-dataset")
        k_paths = []
        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    k_paths.append(os.path.join(root, file))
        if k_paths:
            return sorted(k_paths)[:num_images]
    except Exception as e:
        print(f"Could not download dataset ({e}). Generating synthetic photographic test images...")

    # Generate synthetic photographic images for offline test
    paths = []
    rng = np.random.default_rng(42)
    for idx in range(num_images):
        p = os.path.join(image_dir, f"kodim_synthetic_{idx+1:02d}.png")
        # Create a gradient + shapes test image (512 x 768 x 3)
        h, w = 512, 768
        xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
        r = (np.sin(xx * 3.14) * 255).astype(np.uint8)
        g = (np.cos(yy * 3.14) * 255).astype(np.uint8)
        b = ((xx + yy) * 127.5).astype(np.uint8)
        img_arr = np.stack([r, g, b], axis=-1)
        # Add random patch
        patch = rng.integers(0, 256, (100, 100, 3), dtype=np.uint8)
        img_arr[150:250, 250:350] = patch
        Image.fromarray(img_arr).save(p)
        paths.append(p)
    return paths

def image_to_pixels(image):
    return image.reshape(-1, 3).astype(np.float32)

def pixels_to_image(pixels, shape):
    return pixels.reshape(shape).clip(0, 255).astype(np.uint8)

def fit_and_predict_subsampled(model, pixels, max_samples=50000):
    if len(pixels) > max_samples:
        indices = np.random.choice(len(pixels), max_samples, replace=False)
        model.fit(pixels[indices])
    else:
        model.fit(pixels)
    return model.predict(pixels)

def quantize_kmeans(image, k):
    original_shape = image.shape
    pixels = image_to_pixels(image)
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=3, random_state=42)
    labels = fit_and_predict_subsampled(kmeans, pixels)
    palette = np.round(kmeans.cluster_centers_).astype(np.uint8)
    quantized_pixels = palette[labels]
    quantized_image = pixels_to_image(quantized_pixels, original_shape)
    return quantized_image, palette, labels, kmeans.inertia_

def compute_mse(original, quantized):
    return np.mean((original.astype(np.float64) - quantized.astype(np.float64)) ** 2)

def compute_psnr(original, quantized):
    mse = compute_mse(original, quantized)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))

def compute_ssim(original, quantized):
    return skimage.metrics.structural_similarity(original, quantized, channel_axis=2, data_range=255)

def compute_compression_metrics(shape, k):
    h, w, _ = shape
    original_bits = h * w * 24
    index_bits = h * w * int(np.ceil(np.log2(k)))
    palette_bits = k * 24
    compressed_bits = index_bits + palette_bits
    ratio = original_bits / compressed_bits
    bpp = compressed_bits / (h * w)
    return ratio, bpp

def compute_edge_f1(original, quantized):
    gray_orig = color.rgb2gray(original)
    gray_q = color.rgb2gray(quantized)
    edge_orig = feature.canny(gray_orig, sigma=1.0)
    edge_q = feature.canny(gray_q, sigma=1.0)
    tp = np.sum(edge_orig & edge_q)
    fp = np.sum(~edge_orig & edge_q)
    fn = np.sum(edge_orig & ~edge_q)
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0

def run_unit_3(k_list=(4, 8, 16, 32, 64), output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img_paths = get_sample_images()
    test_path = img_paths[0]
    raw_img = np.array(Image.open(test_path).convert("RGB"))

    print("=" * 70)
    print(f"STEP 1: Loaded Image {os.path.basename(test_path)} (Shape: {raw_img.shape})")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("STEP 2: Palette Size Sweep & Rate-Distortion Evaluation")
    print("=" * 70)
    sweep_results = []
    for k in k_list:
        t0 = time.perf_counter()
        q_img, palette, labels, inertia = quantize_kmeans(raw_img, k)
        elapsed = time.perf_counter() - t0

        mse = compute_mse(raw_img, q_img)
        psnr = compute_psnr(raw_img, q_img)
        ssim = compute_ssim(raw_img, q_img)
        ratio, bpp = compute_compression_metrics(raw_img.shape, k)
        edge_f1 = compute_edge_f1(raw_img, q_img)

        sweep_results.append({
            "K (Colors)": k,
            "BPP": round(bpp, 2),
            "Ratio": round(ratio, 2),
            "PSNR (dB)": round(psnr, 2),
            "SSIM": round(ssim, 4),
            "Edge F1": round(edge_f1, 4),
            "Inertia (x1e7)": round(inertia / 1e7, 3),
            "Time (s)": round(elapsed, 3)
        })

    df_sweep = pd.DataFrame(sweep_results)
    print(df_sweep.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 3: Comparison of Quantization Algorithms at K = 16")
    print("=" * 70)
    k_eval = 16
    comp_results = []

    # 1. K-Means RGB
    t0 = time.perf_counter()
    q_km, _, _, _ = quantize_kmeans(raw_img, k_eval)
    t_km = time.perf_counter() - t0
    comp_results.append({
        "Method": "K-Means (RGB)",
        "PSNR (dB)": round(compute_psnr(raw_img, q_km), 2),
        "SSIM": round(compute_ssim(raw_img, q_km), 4),
        "Time (s)": round(t_km, 3)
    })

    # 2. MiniBatchKMeans RGB
    t0 = time.perf_counter()
    mb = MiniBatchKMeans(n_clusters=k_eval, init="k-means++", n_init=3, random_state=42, batch_size=2048)
    mb_labels = fit_and_predict_subsampled(mb, image_to_pixels(raw_img))
    q_mb = pixels_to_image(np.round(mb.cluster_centers_).astype(np.uint8)[mb_labels], raw_img.shape)
    t_mb = time.perf_counter() - t0
    comp_results.append({
        "Method": "MiniBatchKMeans (RGB)",
        "PSNR (dB)": round(compute_psnr(raw_img, q_mb), 2),
        "SSIM": round(compute_ssim(raw_img, q_mb), 4),
        "Time (s)": round(t_mb, 3)
    })

    # 3. K-Means CIELAB
    t0 = time.perf_counter()
    lab_pixels = color.rgb2lab(raw_img).reshape(-1, 3).astype(np.float32)
    km_lab = KMeans(n_clusters=k_eval, init="k-means++", n_init=3, random_state=42)
    lab_labels = fit_and_predict_subsampled(km_lab, lab_pixels)
    q_lab = np.round(color.lab2rgb(km_lab.cluster_centers_[lab_labels].reshape(raw_img.shape)) * 255).astype(np.uint8)
    t_lab = time.perf_counter() - t0
    comp_results.append({
        "Method": "K-Means (CIELAB)",
        "PSNR (dB)": round(compute_psnr(raw_img, q_lab), 2),
        "SSIM": round(compute_ssim(raw_img, q_lab), 4),
        "Time (s)": round(t_lab, 3)
    })

    # 4. Median-Cut (Pillow)
    t0 = time.perf_counter()
    pil_img = Image.fromarray(raw_img)
    q_mc = np.array(pil_img.quantize(colors=k_eval, method=Image.Quantize.MEDIANCUT).convert("RGB"))
    t_mc = time.perf_counter() - t0
    comp_results.append({
        "Method": "Median-Cut (Pillow)",
        "PSNR (dB)": round(compute_psnr(raw_img, q_mc), 2),
        "SSIM": round(compute_ssim(raw_img, q_mc), 4),
        "Time (s)": round(t_mc, 3)
    })

    df_comp = pd.DataFrame(comp_results)
    print(df_comp.to_string(index=False))

    print("\n" + "=" * 70)
    print("STEP 4: Real Disk Storage / Transmission Benchmark")
    print("=" * 70)
    p_orig_png = os.path.join(output_dir, "original_raw.png")
    p_orig_jpg = os.path.join(output_dir, "original_compressed.jpg")
    p_quant_png = os.path.join(output_dir, "quantized_indexed.png")

    pil_img.save(p_orig_png, "PNG")
    pil_img.save(p_orig_jpg, "JPEG", quality=90)
    Image.fromarray(q_km).convert("P", palette=Image.Palette.ADAPTIVE, colors=k_eval).save(p_quant_png, "PNG")

    s_orig = os.path.getsize(p_orig_png) / 1024
    s_jpg = os.path.getsize(p_orig_jpg) / 1024
    s_quant = os.path.getsize(p_quant_png) / 1024

    print(f"Original 24-bit PNG : {s_orig:7.2f} KB (100.0%)")
    print(f"JPEG (Quality = 90) : {s_jpg:7.2f} KB ({s_jpg/s_orig*100:5.1f}%)")
    print(f"16-Color Indexed PNG: {s_quant:7.2f} KB ({s_quant/s_orig*100:5.1f}%) [Saving: {100 - s_quant/s_orig*100:.1f}%]")

    return df_sweep

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit 3 Colour Quantization")
    args = parser.parse_args()
    run_unit_3()

