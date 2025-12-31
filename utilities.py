import os
from pathlib import Path
from typing import Optional
import torch
from safetensors.torch import save_file


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
            font_name = os.path.splitext(os.path.basename(font_path))[0]
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
