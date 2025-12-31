import os
import subprocess
import itertools
from pathlib import Path
import argparse

# --- 1. CONFIGURATION ---
# Setup argument parser
parser = argparse.ArgumentParser(
    description="Run an ablation experiment for FontDiffuser."
)
parser.add_argument(
    "--content_image_path",
    type=str,
    default="data_examples/content.png",
    help="Path to the content image.",
)
parser.add_argument(
    "--style_image_path",
    type=str,
    default="data_examples/style.png",
    help="Path to the style image.",
)

args, unknown = parser.parse_known_args()

CONTENT_IMAGE_PATH = args.content_image_path
STYLE_IMAGE_PATH = args.style_image_path

# Output Directory
OUTPUT_DIR = "outputs_ablation"

# --- 2. DEFINE PARAMETERS TO TEST ---
# "Guidance Scale": Higher = Forces the style more strictly. Lower = More creative/random.
scales = [i + 0.5 for i in range(5, 10)]

# "Inference Steps": Higher = Cleaner, more detailed strokes. Lower = Faster, rougher.
steps = range(20, 151, 10)

# --- 3. SETUP ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
content_basename = Path(CONTENT_IMAGE_PATH).stem
style_basename = Path(STYLE_IMAGE_PATH).stem

# Generate all possible combinations
combinations = list(itertools.product(scales, steps))

print(f"🚀 Starting Ablation Study: {len(combinations)} experiments")
print(f"📂 Inputs: {content_basename} + {style_basename}")
print(f"📂 Output Directory: {OUTPUT_DIR}")

# --- 4. MAIN LOOP ---
for i, (scale, step) in enumerate(combinations):
    print(f"\n[{i + 1}/{len(combinations)}] Testing: Scale={scale}, Steps={step}...")

    # Construct the command
    cmd = [
        "python",
        "font_diffusion/sample.py",
        "--ckpt_dir",
        "ckpt/",
        "--content_image_path",
        CONTENT_IMAGE_PATH,
        "--style_image_path",
        STYLE_IMAGE_PATH,
        "--save_image",
        "--save_image_dir",
        OUTPUT_DIR,
        "--device",
        "cuda:0",
        "--algorithm_type",
        "dpmsolver++",
        "--guidance_type",
        "classifier-free",
        "--method",
        "multistep",
        # Dynamic Parameters
        "--guidance_scale",
        str(scale),
        "--num_inference_steps",
        str(step),
    ]

    try:
        # Run the generation script
        subprocess.run(cmd, check=True)

        # --- 5. RENAME OUTPUTS ---
        # Define meaningful filenames
        # Format: result_scale-7.5_step-50.png
        filename_suffix = f"scale-{scale}_step-{step}"

        target_single = os.path.join(OUTPUT_DIR, f"result_{filename_suffix}.png")
        target_compare = os.path.join(OUTPUT_DIR, f"compare_{filename_suffix}.png")

        # Default outputs from sample.py
        src_single = os.path.join(OUTPUT_DIR, "out_single.png")
        src_compare = os.path.join(OUTPUT_DIR, "out_with_cs.png")

        # Rename "Single Result"
        if os.path.exists(src_single):
            os.rename(src_single, target_single)
            print(f"   ✅ Saved: {os.path.basename(target_single)}")
        else:
            print("   ⚠️ Warning: out_single.png not found.")

        # Rename "Comparison Grid" (Content | Style | Result)
        if os.path.exists(src_compare):
            os.rename(src_compare, target_compare)

    except subprocess.CalledProcessError:
        print(f"   ❌ Error running experiment {filename_suffix}")
    except Exception as e:
        print(f"   ⚠️ Unexpected error: {e}")

print("\n🎉 Ablation study complete!")
"""Example
!python ablation.py \
  --content_image_path /content/content.jpg \
  --style_image_path /content/sinonom_diffuser/figures/ref_imgs/ref_雕.jpg
"""
