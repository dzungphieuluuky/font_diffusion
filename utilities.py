import os
from pathlib import Path
from typing import Optional
import torch
from safetensors.torch import save_file
import shutil
from tqdm.auto import tqdm
from typing import Any, Dict
import logging
import json
import os

class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[TqdmLoggingHandler()],
)

HF_BLUE = "#05339C"  # Color for active bars
HF_GREEN = "#41A67E" # Color for completed bars

# The format string to match your image exactly:
HF_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

class HFTqdm(tqdm):
    """
    A custom TQDM bar that replicates the Hugging Face download look.
    """
    def __init__(self, *args, **kwargs):
        # Set defaults to match the image
        kwargs.setdefault("unit", "B")
        kwargs.setdefault("unit_scale", True)
        kwargs.setdefault("unit_divisor", 1024)
        kwargs.setdefault("bar_format", HF_BAR_FORMAT)
        kwargs.setdefault("colour", HF_BLUE) # Start as Blue
        kwargs.setdefault("ascii", False)
        kwargs.setdefault("ncols", 64)      # Fixed width
        super().__init__(*args, **kwargs)

    def close(self):
        """Change color to green upon completion."""
        self.colour = HF_GREEN
        self.refresh()
        super().close()

def get_hf_bar(iterable=None, desc="File", total=None, **kwargs):
    """Factory function for the progress bar."""
    return HFTqdm(iterable=iterable, desc=desc, total=total, **kwargs)


def load_model_checkpoint(checkpoint_path: str):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file as safe_load
        state_dict = safe_load(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu")
    return state_dict


def save_model_checkpoint(model_state_dict, checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import save_file as safe_save
        safe_save(model_state_dict, checkpoint_path)
    else:
        torch.save(model_state_dict, checkpoint_path)


def find_checkpoint(checkpoint_dir: str, checkpoint_name: str) -> str:
    """
    Find checkpoint file, preferring .safetensors over .pth
    Args:
        checkpoint_dir: Directory containing checkpoint
        checkpoint_name: Checkpoint name without extension (e.g., "unet")
    Returns:
        Full path to checkpoint file
    Raises:
        FileNotFoundError: If neither format exists
    """
    safetensors_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.safetensors")
    pth_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pth")
    
    if os.path.exists(safetensors_path):
        return safetensors_path
    elif os.path.exists(pth_path):
        return pth_path
    else:
        raise FileNotFoundError(
            f"Checkpoint not found for '{checkpoint_name}' in {checkpoint_dir}\n"
            f"  Expected: {safetensors_path} or {pth_path}"
        )
    
def rename_images(json_file):
    # Load the JSON data
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    generations = data.get("generations", [])

    for entry in generations:
        char = entry.get("character")
        style = entry.get("style")

        # New base name based on your example: style+char.png
        new_filename = f"{style}+{char}.png"

        # List of paths to process for each entry
        paths_to_update = ["content_image_path", "target_image_path"]

        for path_key in paths_to_update:
            old_path = entry.get(path_key)

            if old_path and os.path.exists(old_path):
                # Extract directory (e.g., ContentImage/ or TargetImage/1/)
                directory = os.path.dirname(old_path)

                if "content" in path_key:
                    new_filename = f"{char}.png"
                new_path = os.path.join(directory, new_filename)

                # Rename the actual file on disk
                try:
                    os.rename(old_path, new_path)
                    print(f"Renamed: {old_path} -> {new_path}")
                    # Update the path in the JSON object
                    entry[path_key] = new_path
                except OSError as e:
                    print(f"Error renaming {old_path}: {e}")
            else:
                print(f"Skipping: File not found {old_path}")

    # Save the updated JSON back to a file
    output_file = "updated_generations.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nProcessing complete. Updated JSON saved as: {output_file}")


def rename_content_images(path: str):
    # Rename all files in ContentImage/ from 1+char.png to char.png
    content_dir = os.path.join(path, "ContentImage")
    for filename in os.listdir(content_dir):
        if "+" in filename:
            char = filename.split("+")[1]
            old_path = os.path.join(content_dir, filename)
            new_path = os.path.join(content_dir, char)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")


def update_paths(input_file, output_file=None):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("generations", [])

    for item in items:
        char, style = item["character"], item["style"]
        item["content_image_path"] = f"ContentImage/{char}.png"
        item["target_image_path"] = f"TargetImage/{style}/{style}+{char}.png"

    with open(output_file or input_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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

    parser.add_argument(
        "--rename_images_json",
        type=str,
        help="Path to JSON file for renaming images based on character and style",
    )

    parser.add_argument(
        "--rename_content_images_dir",
        type=str,
        help="Path to directory to rename content images from '1+char.png' to 'char.png'",
    )

    parser.add_argument(
        "--update_paths",
        type=str,
        nargs=2,
        help="Path to JSON file to update content and target image paths",
    )
    args = parser.parse_args()

    if args.flatten_dir:
        flatten_folder(args.flatten_dir)
        print(f"✓ Flattened directory: {args.flatten_dir}")

    if args.font_glyphs_dir:
        print_font_glyph_counts(args.font_glyphs_dir)

    if args.convert_ckpt_dir:
        convert_checkpoint_folder(args.convert_ckpt_dir, args.output_dir)

    if args.rename_images_json:
        rename_images(args.rename_images_json)

    if args.rename_content_images_dir:
        rename_content_images(args.rename_content_images_dir)

    if args.update_paths:
        update_paths(args.update_paths[0], args.update_paths[1])
        print(f"✓ Updated paths in JSON file: {args.update_paths}")

    else:
        print("No arguments provided. Use --help for usage information.")
