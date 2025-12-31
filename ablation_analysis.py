import os
import torch
import lpips
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import re

# --- CONFIG ---
results_folder = "outputs_ablation"
content_ref_path = "/content/content.jpg"
style_ref_path = "/content/font_diffusion/figures/ref_imgs/ref_媚.jpg"

# Load LPIPS Metric
loss_fn_alex = lpips.LPIPS(net="alex").cuda()


def load_tensor(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = (img / 255.0) * 2 - 1
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).cuda()


def calculate_ssim(img_path1, img_path2):
    i1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    i2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    if i1 is None or i2 is None:
        return 0.0
    i1 = cv2.resize(i1, (128, 128))
    i2 = cv2.resize(i2, (128, 128))
    return ssim(i1, i2)


# --- DATA COLLECTION ---
print(f"{'Filename':<45} | {'SSIM':<10} | {'LPIPS':<10}")
print("-" * 75)

tensor_style = load_tensor(style_ref_path)

# Dictionary to store data: data[scale] = {step: (ssim, lpips)}
plot_data = {}

files = sorted(
    [f for f in os.listdir(results_folder) if f.endswith(".png") and "result_" in f]
)

for f in files:
    gen_path = os.path.join(results_folder, f)

    # 1. Calculate Metrics
    score_ssim = calculate_ssim(content_ref_path, gen_path)

    tensor_gen = load_tensor(gen_path)
    if tensor_gen is not None:
        with torch.no_grad():
            score_lpips = loss_fn_alex(tensor_gen, tensor_style).item()
    else:
        score_lpips = 1.0  # Max error if file fail

    print(f"{f:<45} | {score_ssim:.4f}     | {score_lpips:.4f}")

    # 2. Parse Filename to get Parameters
    # Expected format: result_scale-7.5_step-50.png
    try:
        # Regex to extract numbers from "scale-X.X" and "step-XX"
        scale_match = re.search(r"scale-([\d\.]+)", f)
        step_match = re.search(r"step-([\d]+)", f)

        if scale_match and step_match:
            scale = float(scale_match.group(1))
            step = int(step_match.group(1))

            # Store data
            if scale not in plot_data:
                plot_data[scale] = {"steps": [], "ssim": [], "lpips": []}

            plot_data[scale]["steps"].append(step)
            plot_data[scale]["ssim"].append(score_ssim)
            plot_data[scale]["lpips"].append(score_lpips)
    except Exception as e:
        print(f"⚠️ Could not parse parameters from {f}")

# --- VISUALIZATION ---
print("\n📊 Generating Graphs...")

# Make figure slightly wider to accommodate the external legend
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sorted_scales = sorted(plot_data.keys())

for scale in sorted_scales:
    data = plot_data[scale]
    sorted_points = sorted(zip(data["steps"], data["ssim"], data["lpips"]))
    steps, ssims, lpip_scores = zip(*sorted_points)

    # Plot Lines
    ax1.plot(steps, ssims, marker="o", label=f"Guidance {scale}")
    ax2.plot(steps, lpip_scores, marker="o", label=f"Guidance {scale}")

# --- GRAPH 1: SSIM ---
ax1.set_title("Structure Consistency (SSIM)\nHigher is Better ↑")
ax1.set_xlabel("Inference Steps")
ax1.set_ylabel("SSIM Score")
ax1.grid(True, linestyle="--", alpha=0.6)

# LEGEND OUTSIDE RIGHT
ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

# --- GRAPH 2: LPIPS ---
ax2.set_title("Style Similarity (LPIPS)\nLower is Better ↓")
ax2.set_xlabel("Inference Steps")
ax2.set_ylabel("LPIPS Distance")
ax2.grid(True, linestyle="--", alpha=0.6)

# LEGEND OUTSIDE RIGHT
ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)

# Adjust layout to make room for legends
plt.tight_layout()

# Save with bbox_inches='tight' so the legend doesn't get cropped
plt.savefig("ablation_chart.png", bbox_inches="tight", dpi=150)
plt.show()
