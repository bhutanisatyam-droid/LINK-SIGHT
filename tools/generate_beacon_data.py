"""Synthetic Training Data Generator for TinyBeaconNet (Core 1 Vision AI).

Generates 20,000 synthetic 24x24 pixel patches:
- 10,000 Positive Samples (Optical Beacons under PSF, Scintillation, Poisson noise)
- 10,000 Negative Samples (Solar Glints, Dead Pixels, Cloud Edges, Thermal Background)

Also calculates the 8D hand-crafted mathematical feature vector for every patch:
[Brenner, PSLR, Circularity, Radial Symmetry, Skewness, Aspect Ratio, Flux Concentration, Variance]

Output:
Saves dataset to `datasets/dataset_beacon.npz` containing:
- 'patches': float32 array of shape (20000, 1, 24, 24) normalized to [0, 1]
- 'features': float32 array of shape (20000, 8) normalized
- 'labels': float32 array of shape (20000, 1) where 1=Beacon, 0=Glint/Noise
"""

import math
import os
import cv2
import numpy as np
from scipy.stats import skew


def compute_8d_features(patch: np.ndarray) -> np.ndarray:
    """Extract 8-dimensional mathematical descriptor vector from a 24x24 grayscale patch."""
    p = patch.astype(np.float32)
    h, w = p.shape

    # 1. Brenner Sharpness (Horizontal gradient energy)
    diff = p[:, 2:] - p[:, :-2]
    brenner = float(np.sum(diff ** 2)) / (h * w)

    # 2. Peak-to-Sidelobe Ratio (PSLR)
    peak_val = float(np.max(p))
    mask = np.ones((h, w), dtype=bool)
    max_y, max_x = np.unravel_index(np.argmax(p), p.shape)
    y1, y2 = max(0, max_y - 2), min(h, max_y + 3)
    x1, x2 = max(0, max_x - 2), min(w, max_x + 3)
    mask[y1:y2, x1:x2] = False
    sidelobe = p[mask]
    side_mean = float(np.mean(sidelobe))
    side_std = float(np.std(sidelobe)) + 1e-5
    pslr = (peak_val - side_mean) / side_std

    # Threshold for morphological contour analysis
    norm_p = ((p - p.min()) / (p.max() - p.min() + 1e-5) * 255).astype(np.uint8)
    _, thresh = cv2.threshold(norm_p, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        perimeter = float(cv2.arcLength(cnt, True)) + 1e-5
        
        # 3. Circularity: 4*pi*Area / Perimeter^2 (1.0 for perfect circle)
        circularity = min(1.0, 4.0 * math.pi * area / (perimeter ** 2))
        
        # 6. Aspect Ratio of bounding box
        bx, by, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / float(bh + 1e-5)
    else:
        circularity = 0.1
        aspect_ratio = 1.0

    # 4. Radial Symmetry (Cosine similarity of gradient directions vs radial vectors)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    cy, cx = h / 2.0, w / 2.0
    ys, xs = np.ogrid[:h, :w]
    rx = xs - cx
    ry = ys - cy
    r_norm = np.sqrt(rx ** 2 + ry ** 2) + 1e-5
    g_norm = np.sqrt(gx ** 2 + gy ** 2) + 1e-5
    mask_grad = g_norm > 5.0
    dot = np.abs((rx * gx + ry * gy) / (r_norm * g_norm))
    rad_sym = float(np.sum(dot * mask_grad)) / (h * w)

    # 5. Skewness of intensity distribution
    p_mean = np.mean(p)
    p_std = np.std(p) + 1e-5
    skewness = float(np.mean(((p - p_mean) / p_std) ** 3))

    # 7. Flux Concentration (Ratio of energy in center 4x4 vs total energy)
    cy_i, cx_i = h // 2, w // 2
    center_energy = float(np.sum(p[cy_i - 2:cy_i + 2, cx_i - 2:cx_i + 2]))
    total_energy = float(np.sum(p)) + 1e-5
    flux_conc = center_energy / total_energy

    # 8. Local Intensity Variance
    variance = float(np.var(p))

    return np.array([brenner, pslr, circularity, rad_sym, skewness, aspect_ratio, flux_conc, variance], dtype=np.float32)


def generate_beacon_patch() -> np.ndarray:
    """Generate a realistic optical laser beacon patch (24x24) with PSF and noise."""
    cx = 12.0 + np.random.uniform(-2.5, 2.5)
    cy = 12.0 + np.random.uniform(-2.5, 2.5)
    sigma_psf = np.random.uniform(1.2, 2.5)
    peak_intensity = np.random.uniform(180.0, 255.0)
    scintillation = np.random.uniform(0.7, 1.3)

    ys, xs = np.ogrid[:24, :24]
    dist_sq = (xs - cx) ** 2 + (ys - cy) ** 2
    patch = peak_intensity * np.exp(-dist_sq / (2.0 * sigma_psf ** 2)) * scintillation

    # Background ambient floor + Poisson shot noise
    bg_level = np.random.uniform(5.0, 35.0)
    patch += bg_level
    noise = np.random.normal(0, np.sqrt(np.maximum(1.0, patch)) * 0.5)
    return np.clip(patch + noise, 0, 255).astype(np.float32)


def generate_glint_patch() -> np.ndarray:
    """Generate realistic negative non-beacon noise (solar glint, dead pixel, cloud edge)."""
    glint_type = np.random.choice(["solar_streak", "dead_pixel", "diffuse_cloud", "thermal_noise"])
    patch = np.zeros((24, 24), dtype=np.float32)
    ys, xs = np.ogrid[:24, :24]

    if glint_type == "solar_streak":
        angle = np.random.uniform(0, math.pi)
        length = np.random.uniform(6, 14)
        thickness = np.random.uniform(0.6, 1.2)
        cx, cy = 12.0, 12.0
        peak = np.random.uniform(190, 255)
        
        dx = xs - cx
        dy = ys - cy
        u = dx * math.cos(angle) + dy * math.sin(angle)
        v = -dx * math.sin(angle) + dy * math.cos(angle)
        streak_mask = np.abs(u) <= length / 2.0
        val = peak * np.exp(-(v ** 2) / (2.0 * thickness ** 2))
        patch = np.where(streak_mask, val, 0.0)

    elif glint_type == "dead_pixel":
        patch += np.random.uniform(5.0, 20.0)
        px, py = np.random.randint(4, 20), np.random.randint(4, 20)
        patch[py, px] = 255.0

    elif glint_type == "diffuse_cloud":
        grad_dir = np.random.uniform(0, 2 * math.pi)
        slope = np.random.uniform(2.0, 6.0)
        base = np.random.uniform(40.0, 140.0)
        patch = base + slope * (xs * math.cos(grad_dir) + ys * math.sin(grad_dir))

    else: # thermal_noise
        patch = np.random.normal(50.0, 15.0, (24, 24))

    # Add Poisson shot noise
    noise = np.random.normal(0, np.sqrt(np.maximum(1.0, patch)) * 0.7)
    return np.clip(patch + noise, 0, 255).astype(np.float32)


def main():
    print("=" * 60)
    print("LinkSight: TinyBeaconNet Synthetic Dataset Generator")
    print("=" * 60)

    num_samples = 20000
    half = num_samples // 2

    patches = np.zeros((num_samples, 1, 24, 24), dtype=np.float32)
    features = np.zeros((num_samples, 8), dtype=np.float32)
    labels = np.zeros((num_samples, 1), dtype=np.float32)

    print(f"[*] Generating {half} positive optical beacon samples...")
    for i in range(half):
        p = generate_beacon_patch()
        patches[i, 0] = p / 255.0  # Normalize to [0, 1]
        features[i] = compute_8d_features(p)
        labels[i] = 1.0

        if (i + 1) % 2500 == 0:
            print(f"    - Generated {i + 1}/{half} beacon patches")

    print(f"[*] Generating {half} negative solar glint / noise samples...")
    for i in range(half, num_samples):
        p = generate_glint_patch()
        patches[i, 0] = p / 255.0
        features[i] = compute_8d_features(p)
        labels[i] = 0.0

        if (i - half + 1) % 2500 == 0:
            print(f"    - Generated {i - half + 1}/{half} glint/noise patches")

    # Shuffle dataset
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    patches = patches[indices]
    features = features[indices]
    labels = labels[indices]

    # Normalize feature matrix (zero mean, unit variance)
    f_mean = np.mean(features, axis=0)
    f_std = np.std(features, axis=0) + 1e-5
    features_norm = (features - f_mean) / f_std

    os.makedirs("datasets", exist_ok=True)
    out_file = "datasets/dataset_beacon.npz"
    np.savez_compressed(out_file, patches=patches, features=features_norm, labels=labels, f_mean=f_mean, f_std=f_std)

    print(f"\n[SUCCESS] Dataset generated successfully!")
    print(f"[*] Saved to: {out_file}")
    print(f"[*] Total Samples: {num_samples} (Patches: {patches.shape}, Features: {features.shape})")
    print(f"[*] File Size: {os.path.getsize(out_file) / (1024 * 1024):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    main()
