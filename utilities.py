import os
from pathlib import Path
from typing import Optional
import torch
from safetensors.torch import save_file
import shutil
from huggingface_hub.utils import tqdm
from typing import Any, Dict

# ============================================================================
# TQDM CONFIGURATION - Reusable across the file
# ============================================================================

def get_tqdm_config(
    total: Optional[int] = None,
    desc: str = "",
    unit: str = "it",
    ncols: int = 100,
    unit_scale: bool = True,
    unit_divisor: int = 1000,
) -> Dict[str, Any]:
    """
    Get standardized tqdm configuration
    
    ✅ unit_scale=True: Automatically scales to KB, MB, GB, etc.
    ✅ total: Number of items (from loop)
    ✅ Updates automatically as items are processed
    
    Args:
        total: Total number of items to process
        desc: Description/label for the progress bar
        unit: Unit name (default: "it" for items)
        ncols: Width of progress bar (default: 100)
        unit_scale: Enable automatic scaling (default: True)
        unit_divisor: Divisor for unit scaling (1000 for decimal, 1024 for binary)
    
    Returns:
        Dict of tqdm configuration that can be unpacked with **
    
    Example:
        for item in tqdm(items, **get_tqdm_config(
            total=len(items),
            desc="Processing",
            unit="item"
        )):
            # process item
            pass
    """
    config = {
        "total": total,
        "desc": desc,
        "unit": unit,
        "ncols": ncols,
        "unit_scale": unit_scale,
        "unit_divisor": unit_divisor,
        "leave": True,
        "dynamic_ncols": True,
    }
    return {k: v for k, v in config.items() if v is not None}


# Preset configurations for common scenarios
TQDM_IMAGE_LOADING = {
    "desc": "Loading images",
    "unit": "img",
    "ncols": 100,
    "unit_scale": True,
    "unit_divisor": 1000,
}

TQDM_GENERATION_PAIR = {
    "desc": "Processing pairs",
    "unit": "pair",
    "ncols": 100,
    "unit_scale": True,
    "unit_divisor": 1000,
}

TQDM_FILE_IO = {
    "desc": "Processing files",
    "unit": "file",
    "ncols": 100,
    "unit_scale": True,
    "unit_divisor": 1024,  # Binary divisor for file sizes
}


def flatten_folder(root_dir):
    """
    Moves all files from subdirectories of root_dir into root_dir.
    """
    for subdir, dirs, files in os.walk(root_dir):
        if subdir == root_dir:
            continue  # Skip the root directory itself
        for file in files:
            src = os.path.join(subdir, file)
            dst = os.path.join(root_dir, file)
            # If a file with the same name exists, rename to avoid overwrite
            if os.path.exists(dst):
                base, ext = os.path.splitext(file)
                i = 1
                while os.path.exists(os.path.join(root_dir, f"{base}_{i}{ext}")):
                    i += 1
                dst = os.path.join(root_dir, f"{base}_{i}{ext}")
            shutil.move(src, dst)
    # Optionally, remove empty subdirectories
    for subdir, dirs, files in os.walk(root_dir, topdown=False):
        if subdir != root_dir and not os.listdir(subdir):
            os.rmdir(subdir)


def print_font_glyph_counts(fonts_dir: str):
    """
    Print the total number of glyphs (supported Unicode characters) for each font in fonts_dir.
    """
    from fontTools.ttLib import TTFont
    import glob
    import os

    font_extensions = (".ttf", ".otf", ".TTF", ".OTF")
    font_files = [
        os.path.join(fonts_dir, f)
        for f in os.listdir(fonts_dir)
        if f.endswith(font_extensions)
    ]

    print(f"\nFont glyph count summary for directory: {fonts_dir}")
    for font_path in font_files:
        try:
            font = TTFont(font_path)
            cmap = font.getBestCmap()
            num_glyphs = len(cmap)
            font_name = os.path.splitext(os.path.basename(font_path))
            print(f"  {font_name}: {num_glyphs} glyphs")
        except Exception as e:
            print(f"  {font_path}: Failed to read ({e})")


def pth_to_safetensors(self, pth_path: str, output_path: str) -> None:
    """Convert .pth checkpoint to .safetensors format"""
    print(f"\nConverting {os.path.basename(pth_path)} to safetensors...")

    try:
        # Load PyTorch checkpoint
        state_dict = torch.load(pth_path, map_location="cpu")

        # Convert to safetensors
        save_file(state_dict, output_path)

        file_size_pth = os.path.getsize(pth_path) / (1024**3)
        file_size_safe = os.path.getsize(output_path) / (1024**3)

        print(f"✓ Converted: {pth_path}")
        print(f"  .pth size:  {file_size_pth:.2f} GB")
        print(f"  .safetensors size: {file_size_safe:.2f} GB")

    except Exception as e:
        print(f"✗ Error converting {pth_path}: {e}")
        raise


def convert_checkpoint_folder(
    self, ckpt_dir: str, output_dir: Optional[str] = None
) -> str:
    """Convert all .pth files in checkpoint folder to .safetensors"""
    if output_dir is None:
        output_dir = ckpt_dir

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("CONVERTING CHECKPOINTS TO SAFETENSORS")
    print("=" * 60)

    pth_files = list(Path(ckpt_dir).glob("*.pth"))

    if not pth_files:
        print(f"⚠ No .pth files found in {ckpt_dir}")
        return output_dir

    print(f"Found {len(pth_files)} .pth files")

    for pth_file in pth_files:
        safe_file = Path(output_dir) / f"{pth_file.stem}.safetensors"
        self.pth_to_safetensors(str(pth_file), str(safe_file))

    print(f"\n✓ Conversion complete!")
    return output_dir

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Utility functions for font diffusion project"
    )
    parser.add_argument(
        "--flatten_dir",
        type=str,
        help="Path to directory to flatten (move all files from subdirs to root)",
    )
    parser.add_argument(
        "--font_glyphs_dir",
        type=str,
        help="Path to directory containing fonts to print glyph counts",
    )
    parser.add_argument(
        "--convert_ckpt_dir",
        type=str,
        help="Path to directory containing .pth checkpoints to convert to .safetensors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for converted safetensors (if not specified, uses input dir)",
    )

    args = parser.parse_args()

    if args.flatten_dir:
        flatten_folder(args.flatten_dir)
        print(f"✓ Flattened directory: {args.flatten_dir}")

    if args.font_glyphs_dir:
        print_font_glyph_counts(args.font_glyphs_dir)

    if args.convert_ckpt_dir:
        convert_checkpoint_folder(args.convert_ckpt_dir, args.output_dir)
