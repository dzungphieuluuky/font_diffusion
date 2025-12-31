"""
Batch sampling and evaluation for FontDiffuser
✅ Uses hash-based file naming with unicode characters
✅ Uses results_checkpoint.json as single source of truth
✅ Checks existing generations to skip already processed (char, style, font) combinations
✅ Supports resuming from any start_line/end_line pair
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from argparse import Namespace, ArgumentParser

from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline

# Import evaluation metrics
try:
    import lpips

    LPIPS_AVAILABLE: bool = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE: bool = False

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE: bool = True
except ImportError:
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
    FID_AVAILABLE: bool = False

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE: bool = True
except ImportError:
    print("Warning: scikit-image not available. Install with: pip install scikit-image")
    SSIM_AVAILABLE: bool = False

try:
    import wandb

    WANDB_AVAILABLE: bool = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE: bool = False

# Import FontDiffuser modules
from sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
)
from utils import load_ttf, ttf2im, is_char_in_font


def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """
    Compute deterministic hash for a (character, style, font) combination

    Args:
        char: Unicode character
        style: Style name
        font: Font name (optional)

    Returns:
        8-character hash string
    """
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def get_content_filename(char: str, font: str = "") -> str:
    """
    Get content image filename for character
    Format: {unicode_codepoint}_{char}_{hash}.png
    Example: U+4E00_中_a1b2c3d4.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, "", font)
    # Sanitize char for filename (replace problematic characters)
    safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""
    if safe_char:
        return f"{codepoint}_{safe_char}_{hash_val}.png"
    else:
        return f"{codepoint}_{hash_val}.png"


def get_target_filename(char: str, style: str, font: str = "") -> str:
    """
    Get target image filename
    Format: {unicode_codepoint}_{char}_{style}_{hash}.png
    Example: U+4E00_中_style0_a1b2c3d4.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, style, font)
    safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""
    if safe_char:
        return f"{codepoint}_{safe_char}_{style}_{hash_val}.png"
    else:
        return f"{codepoint}_{style}_{hash_val}.png"


class FontManager:
    """Manages multiple font files"""

    def __init__(self, ttf_path: str) -> None:
        """
        Initialize font manager

        Args:
            ttf_path: Path to a single font file or directory containing fonts
        """
        self.fonts: Dict[str, Dict[str, Any]] = {}
        self.font_paths: List[str] = []
        self._load_fonts(ttf_path)

    def _load_fonts(self, ttf_path: str) -> None:
        """Load font(s) from path"""
        if os.path.isfile(ttf_path):
            # Single font file
            self.font_paths = [ttf_path]
            font_name: str = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": load_ttf(ttf_path),
                "name": font_name,
            }
            print(f"✓ Loaded font: {font_name}")

        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions: Set[str] = {".ttf", ".otf", ".TTF", ".OTF"}
            font_files: List[str] = [
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_extensions
            ]

            if not font_files:
                raise ValueError(f"No font files found in directory: {ttf_path}")

            self.font_paths = sorted(font_files)

            print(f"\n{'=' * 60}")
            print(f"Loading {len(font_files)} fonts from directory...")
            print("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                try:
                    self.fonts[font_name] = {
                        "path": font_path,
                        "font": load_ttf(font_path),
                        "name": font_name,
                    }
                    print(f"✓ Loaded: {font_name}")
                except Exception as e:
                    print(f"✗ Failed to load {font_name}: {e}")

            print("=" * 60)
            print(f"Successfully loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")

    def get_font_names(self) -> List[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())

    def get_font(self, font_name: str) -> Any:
        """Get font object by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["font"]

    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["path"]

    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font"""
        font_path: str = self.get_font_path(font_name)
        return is_char_in_font(font_path, char)

    def get_available_chars_for_font(
        self, font_name: str, characters: List[str]
    ) -> List[str]:
        """Get list of characters available in specific font"""
        return [char for char in characters if self.is_char_in_font(font_name, char)]


class GenerationTracker:
    """
    ✅ Tracks which (character, style, font) combinations have been generated
    Uses hash-based checking for fast lookups
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize generation tracker

        Args:
            checkpoint_path: Path to results_checkpoint.json file
        """
        self.generated_hashes: Set[str] = set()
        self.generations: List[Dict[str, Any]] = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint"""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            raw_generations = results.get("generations", [])
            
            # ✅ Track duplicates
            seen_hashes: Set[str] = set()
            unique_generations: List[Dict[str, Any]] = []
            duplicate_count: int = 0

            # Build hash set for fast lookup and deduplicate
            for gen in raw_generations:
                target_hash = gen.get("target_hash")
                
                if not target_hash:
                    # Compute hash if not in checkpoint
                    char = gen.get("character", "")
                    style = gen.get("style", "")
                    font = gen.get("font", "")
                    
                    # Skip invalid entries
                    if not char or not style:
                        continue
                        
                    target_hash = compute_file_hash(char, style, font)
                
                # ✅ Check for duplicates
                if target_hash in seen_hashes:
                    duplicate_count += 1
                    continue  # Skip duplicate
                
                # Add to collections
                seen_hashes.add(target_hash)
                self.generated_hashes.add(target_hash)
                unique_generations.append(gen)
            
            # ✅ Store only unique generations
            self.generations = unique_generations

            print(f"✓ Loaded checkpoint: {len(self.generations)} unique generations")
            if duplicate_count > 0:
                print(f"  ⚠️  Removed {duplicate_count} duplicate entries")
            print(f"  Total raw entries: {len(raw_generations)}")

        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            import traceback
            traceback.print_exc()

    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if (char, style, font) combination has been generated"""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes

    def mark_generated(self, char: str, style: str, font: str = "") -> None:
        """Mark a (char, style, font) combination as generated"""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)

    def add_generation(self, generation: Dict[str, Any]) -> None:
        """Add a generation record"""
        self.generations.append(generation)

        # Also add to hash set
        char = generation.get("character", "")
        style = generation.get("style", "")
        font = generation.get("font", "")
        self.mark_generated(char, style, font)


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID"""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device: str = device

        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn: Optional[Any] = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn: Optional[Any] = None

        self.transform_to_tensor: transforms.ToTensor = transforms.ToTensor()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0

        try:
            # Convert to tensors [-1, 1]
            img1_tensor: torch.Tensor = (
                self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
            )
            img2_tensor: torch.Tensor = (
                self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
            )

            with torch.inference_mode():
                lpips_value: float = self.lpips_fn(img1_tensor, img2_tensor).item()

            return lpips_value
        except Exception as e:
            print(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0

        try:
            # Convert to grayscale numpy arrays
            img1_gray: np.ndarray = np.array(img1.convert("L"))
            img2_gray: np.ndarray = np.array(img2.convert("L"))

            ssim_value: float = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            return -1.0

    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images"""
        if not FID_AVAILABLE:
            return -1.0

        try:
            fid_value: float = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir], batch_size=50, device=self.device, dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"Error computing FID: {e}")
            return -1.0

    def save_image(self, image: Image.Image, path: str) -> None:
        """Save PIL image to path"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)
        except Exception as e:
            print(f"Error saving image to {path}: {e}")


def parse_args() -> Namespace:
    """Parse command line arguments"""
    parser: ArgumentParser = argparse.ArgumentParser(
        description="Batch sampling and evaluation"
    )

    # Input/Output
    parser.add_argument(
        "--characters",
        type=str,
        required=True,
        help="Comma-separated list of characters or path to text file",
    )
    parser.add_argument(
        "--start_line",
        type=int,
        default=1,
        help="Start line number for character file (1-indexed)",
    )
    parser.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="End line number for character file (inclusive, None = end of file)",
    )
    parser.add_argument(
        "--style_images",
        type=str,
        required=True,
        help="Comma-separated paths to style images or directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_examples/train_original",
        help="Output directory (will create ContentImage/ and TargetImage/ subdirs)",
    )
    parser.add_argument(
        "--ground_truth_dir",
        type=str,
        default=None,
        help="Directory with ground truth images for evaluation",
    )

    # Model configuration
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument(
        "--ttf_path",
        type=str,
        required=True,
        help="Path to TTF font file or directory with multiple fonts",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    # Generation parameters
    parser.add_argument(
        "--num_inference_steps", type=int, default=15, help="Number of inference steps"
    )
    parser.add_argument(
        "--guidance_scale", type=float, default=7.5, help="Guidance scale"
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for generation"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Optimization flags
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="Use FP16 precision"
    )
    parser.add_argument(
        "--compile", action="store_true", default=False, help="Use torch.compile"
    )
    parser.add_argument(
        "--channels_last",
        action="store_true",
        default=True,
        help="Use channels last memory format",
    )
    parser.add_argument(
        "--enable_xformers", action="store_true", default=False, help="Enable xformers"
    )
    parser.add_argument(
        "--fast_sampling",
        action="store_true",
        default=False,
        help="Use fast sampling mode",
    )

    # Checkpoint and resume
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="Save results every N styles (0 = only save at end)",
    )

    # Evaluation flags
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate generated images",
    )
    parser.add_argument(
        "--compute_fid",
        action="store_true",
        default=False,
        help="Compute FID (requires ground truth)",
    )
    parser.add_argument(
        "--enable_attention_slicing",
        action="store_true",
        default=False,
        help="Enable attention slicing for memory efficiency",
    )

    # Wandb configuration
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
        help="Log results to Weights & Biases",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fontdiffuser-eval",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Wandb run name"
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train_original",
        help="Dataset split name (e.g., train_original, val)",
    )

    return parser.parse_args()


def load_characters(
    characters_arg: str, start_line: int = 1, end_line: Optional[int] = None
) -> List[str]:
    """Load characters from file or comma-separated string with line range support"""
    chars: List[str] = []
    if os.path.isfile(characters_arg):
        with open(characters_arg, "r", encoding="utf-8") as f:
            all_lines: List[str] = f.readlines()

        # Adjust for 1-indexed input
        start_idx: int = max(0, start_line - 1)
        end_idx: int = (
            len(all_lines) if end_line is None else min(len(all_lines), end_line)
        )

        # ✅ ADD VALIDATION
        if start_idx >= len(all_lines):
            raise ValueError(
                f"❌ start_line ({start_line}) exceeds file length ({len(all_lines)} lines)\n"
                f"   Your file only has {len(all_lines)} lines, but you're trying to start at line {start_line}."
            )

        if start_idx >= end_idx:
            raise ValueError(
                f"❌ Invalid line range: start_line={start_line}, end_line={end_line}\n"
                f"   File has {len(all_lines)} lines.\n"
                f"   Computed range [{start_idx}:{end_idx}] is empty.\n"
                f"   Make sure start_line <= end_line and both are within file bounds."
            )

        print(f"📖 Loading characters from file: {characters_arg}")
        print(
            f"   Lines {start_line} to {end_idx} (total file: {len(all_lines)} lines)"
        )
        print(f"   Processing {end_idx - start_idx} lines...")

        for line_num, line in tqdm(
            enumerate(all_lines[start_idx:end_idx], start=start_line),
            total=(end_idx - start_idx),
            desc="📖 Reading character file",
            ncols=100,
            unit="line",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            colour="cyan",
        ):
            char: str = line.strip()
            if not char:
                continue
            if len(char) != 1:
                tqdm.write(
                    f"Warning: Skipping line {line_num}: expected 1 char, got {len(char)}: '{char}'"
                )
                continue
            chars.append(char)

    else:
        for c in [x.strip() for x in characters_arg.split(",") if x.strip()]:
            if len(c) != 1:
                raise ValueError(
                    f"Invalid character in argument: '{c}' (must be single char)"
                )
            chars.append(c)

    # ✅ ADD FINAL CHECK
    if not chars:
        raise ValueError(
            f"❌ No valid characters loaded!\n"
            f"   Check your character file or line range (start={start_line}, end={end_line})"
        )

    print(f"✅ Successfully loaded {len(chars)} single characters.")
    return chars


def load_style_images(style_images_arg: str) -> List[Tuple[str, str]]:
    """
    Load style image paths and extract style names
    Returns: List of (style_path, style_name) tuples
    """
    if os.path.isdir(style_images_arg):
        # Load all images from directory
        image_exts: Set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
        style_paths: List[str] = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()

        print(f"\n📂 Loading {len(style_paths)} style images from directory...")
        verified_paths = []
        for path in tqdm(
            style_paths,
            desc="✓ Verifying style images",
            ncols=100,
            unit="image",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]",
            colour="green",
        ):
            if os.path.isfile(path):
                # Extract style name from filename (without extension)
                style_name = os.path.splitext(os.path.basename(path))[0]
                verified_paths.append((path, style_name))

        return verified_paths
    else:
        style_paths: List[str] = [p.strip() for p in style_images_arg.split(",")]
        result = []
        for path in style_paths:
            style_name = os.path.splitext(os.path.basename(path))[0]
            result.append((path, style_name))
        return result


def create_args_namespace(args: Namespace) -> Namespace:
    """Create args namespace for FontDiffuser pipeline"""

    try:
        from configs.fontdiffuser import get_parser

        parser: ArgumentParser = get_parser()
        default_args: Namespace = parser.parse_args([])
    except Exception:
        default_args: Namespace = Namespace()

    # Override with user arguments
    for key, value in vars(args).items():
        setattr(default_args, key, value)

    # Ensure image sizes are tuples
    if not hasattr(default_args, "style_image_size"):
        default_args.style_image_size = (96, 96)
    elif isinstance(default_args.style_image_size, int):
        default_args.style_image_size = (
            default_args.style_image_size,
            default_args.style_image_size,
        )

    if not hasattr(default_args, "content_image_size"):
        default_args.content_image_size = (96, 96)
    elif isinstance(default_args.content_image_size, int):
        default_args.content_image_size = (
            default_args.content_image_size,
            default_args.content_image_size,
        )

    # Set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    default_args.resolution = 96

    # Generation parameters
    default_args.algorithm_type = getattr(default_args, "algorithm_type", "dpmsolver++")
    default_args.guidance_type = getattr(
        default_args, "guidance_type", "classifier-free"
    )
    default_args.method = getattr(default_args, "method", "multistep")
    default_args.order = getattr(default_args, "order", 2)
    default_args.model_type = getattr(default_args, "model_type", "noise")
    default_args.t_start = getattr(default_args, "t_start", 1.0)
    default_args.t_end = getattr(default_args, "t_end", 1e-3)
    default_args.skip_type = getattr(default_args, "skip_type", "time_uniform")
    default_args.correcting_x0_fn = getattr(default_args, "correcting_x0_fn", None)
    default_args.content_encoder_downsample_size = getattr(
        default_args, "content_encoder_downsample_size", 3
    )

    return default_args


def save_checkpoint(results: Dict[str, Any], output_dir: str) -> None:
    """
    ✅ Save results_checkpoint.json (single source of truth)
    """
    try:
        checkpoint_path: str = os.path.join(output_dir, "results_checkpoint.json")

        # Ensure metrics exist
        if "metrics" not in results:
            results["metrics"] = {"lpips": [], "ssim": [], "inference_times": []}

        # Save checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        num_gens = len(results.get("generations", []))
        print(f"  ✅ Saved results_checkpoint.json ({num_gens} generations)")

    except Exception as e:
        print(f"  ⚠ Error saving checkpoint: {e}")


def generate_content_images(
    characters: List[str],
    font_manager: FontManager,
    output_dir: str,
    generation_tracker: GenerationTracker,
) -> Dict[str, str]:
    """
    Generate and save content character images
    Returns: char_paths dict mapping character to file path
    """
    content_dir: str = os.path.join(output_dir, "ContentImage")
    os.makedirs(content_dir, exist_ok=True)

    font_names: List[str] = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")

    print(f"\n{'=' * 60}")
    print(f"Generating Content Images")
    print(f"Using {len(font_names)} fonts")
    print(f"Characters: {len(characters)}")
    print("=" * 60)

    char_paths: Dict[str, str] = {}
    chars_without_fonts: List[str] = []

    for char in tqdm(
        characters,
        desc="📸 Generating content images",
        ncols=120,
        unit="char",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="blue",
        dynamic_ncols=True,
    ):
        found_font = None
        for font_name in font_names:
            if font_manager.is_char_in_font(font_name, char):
                found_font = font_name
                break

        if not found_font:
            tqdm.write(f"  ⚠ Warning: '{char}' not in any font, skipping...")
            chars_without_fonts.append(char)
            continue

        try:
            font = font_manager.get_font(found_font)
            content_img: Image.Image = ttf2im(font=font, char=char)

            # Use hash-based filename
            content_filename = get_content_filename(char, found_font)
            char_path: str = os.path.join(content_dir, content_filename)

            content_img.save(char_path)
            char_paths[char] = char_path
        except Exception as e:
            tqdm.write(f"  ✗ Error generating '{char}': {e}")

    print(f"✓ Generated {len(char_paths)} content images")
    if chars_without_fonts:
        print(f"⚠ {len(chars_without_fonts)} characters not found in any font")
    print("=" * 60)

    return char_paths


def batch_generate_images(
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_paths_with_names: List[Tuple[str, str]],
    output_dir: str,
    args: Namespace,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
) -> Dict[str, Any]:
    """
    ✅ Main batch generation with hash-based file naming
    Checks generation_tracker to skip already processed combinations
    """

    # Generate ALL content images first
    print(f"\n{'=' * 70}")
    print(f"{'GENERATING CONTENT IMAGES':^70}")
    print("=" * 70)

    char_paths = generate_content_images(
        characters, font_manager, output_dir, generation_tracker
    )

    if not char_paths:
        raise ValueError("No content images generated!")

    # Initialize results from tracker
    results = {
        "generations": generation_tracker.generations.copy(),
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": args.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": list(char_paths.keys()),
        "styles": [name for _, name in style_paths_with_names],
        "total_chars": len(char_paths),
        "total_styles": len(style_paths_with_names),
    }

    # Setup directories
    target_base_dir = os.path.join(output_dir, "TargetImage")
    os.makedirs(target_base_dir, exist_ok=True)

    # Print configuration
    print(f"\n{'=' * 70}")
    print(f"{'BATCH IMAGE GENERATION':^70}")
    print("=" * 70)
    print(f"Fonts:                {len(font_manager.get_font_names())}")
    print(f"Styles:               {len(style_paths_with_names)}")
    print(f"Characters:           {len(characters)}")
    print(f"Batch size:           {args.batch_size}")
    print(f"Existing generations: {len(generation_tracker.generations)} unique pairs")
    print(f"Existing hashes:      {len(generation_tracker.generated_hashes)}")
    print("=" * 70 + "\n")

    # Use first font for all characters
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded!")

    primary_font = font_names[0]
    print(f"Using font: {primary_font}")
    print("=" * 70 + "\n")

    # Initialize counters
    generated_count = 0
    skipped_count = 0
    failed_count = 0
    generation_start_time = time.time()

    # Main generation loop
    for style_idx, (style_path, style_name) in tqdm(
        enumerate(style_paths_with_names),
        total=len(style_paths_with_names),
        desc="🎨 Generating styles",
        ncols=120,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [⏱️ {elapsed}<{remaining}, {rate_fmt}]",
        colour="magenta",
        dynamic_ncols=True,
        position=0,
        leave=True,
    ):
        style_dir = os.path.join(target_base_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)

        try:
            # Filter characters that haven't been generated yet
            chars_to_generate = [
                char
                for char in characters
                if not generation_tracker.is_generated(char, style_name, primary_font)
            ]

            if not chars_to_generate:
                tqdm.write(
                    f"  ⊘ {style_name}: All characters already generated, skipping"
                )
                skipped_count += len(characters)
                continue

            tqdm.write(
                f"  🔄 {style_name}: Generating {len(chars_to_generate)}/{len(characters)} new images"
            )

            # Sample batch
            images, valid_chars, batch_time = sampling_batch_optimized(
                args, pipe, chars_to_generate, style_path, font_manager, primary_font
            )

            if images is None:
                tqdm.write(f"  ⚠️ {style_name}: No images generated")
                skipped_count += len(chars_to_generate)
                continue

            tqdm.write(f"  ✓ {style_name}: {len(images)} images in {batch_time:.2f}s")

            # Save images and metadata
            for char, img in zip(valid_chars, images):
                # Use hash-based filename
                target_filename = get_target_filename(char, style_name, primary_font)
                img_path = os.path.join(style_dir, target_filename)

                content_filename = get_content_filename(char, primary_font)
                content_path_rel = f"ContentImage/{content_filename}"
                target_path_rel = f"TargetImage/{style_name}/{target_filename}"

                evaluator.save_image(img, img_path)

                # Add generation record with hashes
                generation_record = {
                    "character": char,
                    "style": style_name,
                    "font": primary_font,
                    "content_image_path": content_path_rel,
                    "target_image_path": target_path_rel,
                    "content_hash": compute_file_hash(char, "", primary_font),
                    "target_hash": compute_file_hash(char, style_name, primary_font),
                }

                results["generations"].append(generation_record)
                generation_tracker.add_generation(generation_record)
                generated_count += 1

            # Track inference time
            results["metrics"]["inference_times"].append(
                {
                    "style": style_name,
                    "font": primary_font,
                    "total_time": batch_time,
                    "num_images": len(images),
                    "time_per_image": batch_time / len(images) if images else 0,
                }
            )

            # Save checkpoint periodically
            if args.save_interval > 0 and (style_idx + 1) % args.save_interval == 0:
                _print_checkpoint_status(
                    style_idx + 1,
                    len(style_paths_with_names),
                    generated_count,
                    skipped_count,
                    generation_start_time,
                )
                save_checkpoint(results, output_dir)

        except Exception as e:
            tqdm.write(f"  ✗ {style_name}: {e}")
            import traceback

            traceback.print_exc()
            failed_count += len(chars_to_generate)

    # Final statistics
    _print_generation_summary(
        generated_count,
        skipped_count,
        failed_count,
        len(characters) * len(style_paths_with_names),
        generation_start_time,
    )

    return results


def sampling_batch_optimized(
    args: Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_image_path: Union[str, Image.Image],
    font_manager: FontManager,
    font_name: str,
) -> Tuple[Optional[List[Image.Image]], Optional[List[str]], Optional[float]]:
    """Batch sampling for multiple characters with specific font"""

    # Get available characters for this font
    available_chars: List[str] = font_manager.get_available_chars_for_font(
        font_name, characters
    )

    if not available_chars:
        return None, None, None

    try:
        # Load style image
        if isinstance(style_image_path, str):
            style_image: Image.Image = Image.open(style_image_path).convert("RGB")
        else:
            style_image: Image.Image = style_image_path.convert("RGB")
        style_transform: transforms.Compose = get_style_transform(args.style_image_size)

        font: Any = font_manager.get_font(font_name)
        content_transform: transforms.Compose = get_content_transform(
            args.content_image_size
        )

        # Generate content images
        content_images: List[torch.Tensor] = []
        content_images_pil: List[Image.Image] = []

        for char in tqdm(
            available_chars,
            desc=f"  📸 Preparing {font_name}",
            ncols=100,
            unit="char",
            leave=False,
            bar_format="  {desc}: {n_fmt}/{total_fmt} |{bar}| [{elapsed}]",
            colour="cyan",
            position=1,
        ):
            try:
                content_image: Image.Image = ttf2im(font=font, char=char)
                content_images_pil.append(content_image.copy())
                content_images.append(content_transform(content_image))
            except Exception as e:
                tqdm.write(f"    ✗ Error processing '{char}': {e}")
                continue

        if not content_images:
            return None, None, None

        # Stack into batch
        content_batch: torch.Tensor = torch.stack(content_images)
        style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(
            len(content_images), 1, 1, 1
        )

        with torch.inference_mode():
            dtype: torch.dtype = torch.float16 if args.fp16 else torch.float32
            content_batch = content_batch.to(args.device, dtype=dtype)
            style_batch = style_batch.to(args.device, dtype=dtype)

            start: float = time.perf_counter()

            # Process in batches
            all_images: List[Image.Image] = []
            batch_size: int = args.batch_size

            num_batches = (len(content_batch) + batch_size - 1) // batch_size
            batch_pbar = tqdm(
                range(0, len(content_batch), batch_size),
                desc=f"  🎨 Inferencing",
                ncols=100,
                unit="batch",
                ascii=True,
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix[0]} [{elapsed}<{remaining}, {rate_fmt}{postfix[1]}]",
                colour="blue",
                position=2,
            )
            for batch_idx, i in enumerate(batch_pbar):
                batch_content: torch.Tensor = content_batch[i : i + batch_size]
                batch_style: torch.Tensor = style_batch[i : i + batch_size]

                images: List[Image.Image] = pipe.generate(
                    content_images=batch_content,
                    style_images=batch_style,
                    batch_size=len(batch_content),
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn,
                )

                all_images.extend(images)
                batch_pbar.update(1)

            end: float = time.perf_counter()
            total_time: float = end - start

            return all_images, available_chars, total_time

    except Exception as e:
        tqdm.write(f"    ✗ Error in batch sampling: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def _print_checkpoint_status(
    current_style: int,
    total_styles: int,
    generated: int,
    skipped: int,
    start_time: float,
) -> None:
    """Print periodic checkpoint status"""
    elapsed = time.time() - start_time
    remaining = (
        elapsed * (total_styles - current_style) / current_style
        if current_style > 0
        else 0
    )

    print(f"\n{'=' * 70}")
    print(f"{'CHECKPOINT':^70}")
    print("=" * 70)
    print(f"Progress:           {current_style}/{total_styles} styles")
    print(f"Generated:          {generated} pairs")
    print(f"Skipped:            {skipped} pairs")
    print(f"Elapsed time:       {elapsed / 60:.1f} minutes")
    print(f"Est. remaining:     {remaining / 60:.1f} minutes")
    print("=" * 70)


def _print_generation_summary(
    generated: int, skipped: int, failed: int, total: int, start_time: float
) -> None:
    """Print final generation summary"""
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print(f"{'GENERATION COMPLETE':^70}")
    print("=" * 70)
    print(f"\nPair Statistics:")
    print(f"  Total possible:     {total}")
    print(f"  Generated (new):    {generated}")
    print(f"  Skipped (exist):    {skipped}")
    print(f"  Failed (no font):   {failed}")
    print(f"\nTiming:")
    print(f"  Total time:         {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    print(
        f"  Avg per pair:       {elapsed / generated * 1000:.1f}ms"
        if generated > 0
        else "  Avg per pair:       N/A"
    )
    print("=" * 70)


def evaluate_results(
    results: Dict[str, Any],
    evaluator: QualityEvaluator,
    ground_truth_dir: Optional[str] = None,
    compute_fid: bool = False,
) -> Dict[str, Any]:
    """Evaluate generated images against ground truth"""

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        print(
            "\n⚠ No ground truth directory provided or not found, skipping evaluation"
        )
        return results

    print("\n" + "=" * 70)
    print(f"{'EVALUATING GENERATED IMAGES':^70}")
    print("=" * 70)

    lpips_scores: List[float] = []
    ssim_scores: List[float] = []
    evaluated_pairs: int = 0
    missing_gt: int = 0

    # Evaluate each generation
    for gen in tqdm(
        results["generations"],
        desc="📊 Evaluating",
        ncols=100,
        unit="pair",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        colour="green",
    ):
        char: str = gen["character"]
        style: str = gen["style"]
        font: str = gen.get("font", "")

        # Get generated image path
        target_path: str = gen["target_image_path"]
        generated_path: str = os.path.join(
            os.path.dirname(os.path.dirname(target_path)), target_path
        )

        if not os.path.exists(generated_path):
            continue

        # Find ground truth image
        gt_filename = get_target_filename(char, style, font)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            # Try alternative naming
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        try:
            # Load images
            generated_img: Image.Image = Image.open(generated_path).convert("RGB")
            gt_img: Image.Image = Image.open(gt_path).convert("RGB")

            # Compute metrics
            if LPIPS_AVAILABLE:
                lpips_score: float = evaluator.compute_lpips(generated_img, gt_img)
                if lpips_score >= 0:
                    lpips_scores.append(lpips_score)
                    gen["lpips"] = lpips_score

            if SSIM_AVAILABLE:
                ssim_score: float = evaluator.compute_ssim(generated_img, gt_img)
                if ssim_score >= 0:
                    ssim_scores.append(ssim_score)
                    gen["ssim"] = ssim_score

            evaluated_pairs += 1

        except Exception as e:
            tqdm.write(f"  ⚠ Error evaluating {char}/{style}: {e}")
            continue

    # Compute aggregate metrics
    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "min": float(np.min(lpips_scores)),
            "max": float(np.max(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }
        print(f"\n📊 LPIPS Statistics:")
        print(f"  Mean:   {results['metrics']['lpips']['mean']:.4f}")
        print(f"  Std:    {results['metrics']['lpips']['std']:.4f}")
        print(f"  Median: {results['metrics']['lpips']['median']:.4f}")
        print(
            f"  Range:  [{results['metrics']['lpips']['min']:.4f}, {results['metrics']['lpips']['max']:.4f}]"
        )

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "min": float(np.min(ssim_scores)),
            "max": float(np.max(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }
        print(f"\n📊 SSIM Statistics:")
        print(f"  Mean:   {results['metrics']['ssim']['mean']:.4f}")
        print(f"  Std:    {results['metrics']['ssim']['std']:.4f}")
        print(f"  Median: {results['metrics']['ssim']['median']:.4f}")
        print(
            f"  Range:  [{results['metrics']['ssim']['min']:.4f}, {results['metrics']['ssim']['max']:.4f}]"
        )

    # Compute FID if requested
    if compute_fid and FID_AVAILABLE:
        print("\n📊 Computing FID score...")
        try:
            # Create temporary directories for FID computation
            fake_dir = os.path.join(
                os.path.dirname(generated_path), "..", "TargetImage"
            )
            real_dir = os.path.join(ground_truth_dir, "TargetImage")

            if os.path.exists(fake_dir) and os.path.exists(real_dir):
                fid_value: float = evaluator.compute_fid(real_dir, fake_dir)
                if fid_value >= 0:
                    results["metrics"]["fid"] = fid_value
                    print(f"  FID Score: {fid_value:.2f}")
            else:
                print("  ⚠ Cannot compute FID: directories not found")
        except Exception as e:
            print(f"  ⚠ Error computing FID: {e}")

    print("\n" + "=" * 70)
    print(f"{'EVALUATION SUMMARY':^70}")
    print("=" * 70)
    print(f"Evaluated pairs:    {evaluated_pairs}")
    print(f"Missing GT images:  {missing_gt}")
    print(f"LPIPS samples:      {len(lpips_scores)}")
    print(f"SSIM samples:       {len(ssim_scores)}")
    print("=" * 70)

    return results


def log_to_wandb(results: Dict[str, Any], args: Namespace) -> None:
    """Log results to Weights & Biases"""

    if not WANDB_AVAILABLE:
        print("\n⚠ Wandb not available, skipping logging")
        return

    try:
        print("\n" + "=" * 70)
        print(f"{'LOGGING TO WEIGHTS & BIASES':^70}")
        print("=" * 70)

        # Initialize wandb
        run_name = (
            args.wandb_run_name
            or f"{args.dataset_split}_{time.strftime('%Y%m%d_%H%M%S')}"
        )

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset_split": args.dataset_split,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
                "batch_size": args.batch_size,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "fp16": args.fp16,
                "compile": args.compile,
                "xformers": args.enable_xformers,
            },
        )

        # Log generation statistics
        num_generations = len(results.get("generations", []))
        wandb.log(
            {
                "total_generations": num_generations,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
            }
        )

        # Log metrics if available
        metrics = results.get("metrics", {})

        if "lpips" in metrics and isinstance(metrics["lpips"], dict):
            wandb.log(
                {
                    "lpips/mean": metrics["lpips"]["mean"],
                    "lpips/std": metrics["lpips"]["std"],
                    "lpips/median": metrics["lpips"]["median"],
                    "lpips/min": metrics["lpips"]["min"],
                    "lpips/max": metrics["lpips"]["max"],
                }
            )

        if "ssim" in metrics and isinstance(metrics["ssim"], dict):
            wandb.log(
                {
                    "ssim/mean": metrics["ssim"]["mean"],
                    "ssim/std": metrics["ssim"]["std"],
                    "ssim/median": metrics["ssim"]["median"],
                    "ssim/min": metrics["ssim"]["min"],
                    "ssim/max": metrics["ssim"]["max"],
                }
            )

        if "fid" in metrics:
            wandb.log({"fid": metrics["fid"]})

        # Log inference timing
        if "inference_times" in metrics and metrics["inference_times"]:
            timing_data = metrics["inference_times"]

            total_times = [t["total_time"] for t in timing_data if "total_time" in t]
            times_per_image = [
                t["time_per_image"] for t in timing_data if "time_per_image" in t
            ]

            if total_times:
                wandb.log(
                    {
                        "timing/mean_batch_time": np.mean(total_times),
                        "timing/total_time": np.sum(total_times),
                    }
                )

            if times_per_image:
                wandb.log(
                    {
                        "timing/mean_time_per_image": np.mean(times_per_image),
                        "timing/median_time_per_image": np.median(times_per_image),
                    }
                )

        # Log sample images
        print("\n📸 Logging sample images...")
        sample_generations = results.get("generations", [])[:20]  # Log first 20

        sample_images = []
        for gen in sample_generations:
            target_path = gen.get("target_image_path", "")
            if target_path:
                full_path = os.path.join(args.output_dir, target_path)
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path)
                        sample_images.append(
                            wandb.Image(
                                img,
                                caption=f"{gen['character']} - {gen['style']} ({gen.get('font', '')})",
                            )
                        )
                    except Exception as e:
                        print(f"  ⚠ Error loading image {full_path}: {e}")

        if sample_images:
            wandb.log({"sample_images": sample_images})
            print(f"✓ Logged {len(sample_images)} sample images")

        # Create summary table
        generation_table = wandb.Table(
            columns=[
                "Character",
                "Style",
                "Font",
                "LPIPS",
                "SSIM",
                "Content Path",
                "Target Path",
            ]
        )

        for gen in results.get("generations", [])[:100]:  # Log first 100
            generation_table.add_data(
                gen.get("character", ""),
                gen.get("style", ""),
                gen.get("font", ""),
                gen.get("lpips", -1),
                gen.get("ssim", -1),
                gen.get("content_image_path", ""),
                gen.get("target_image_path", ""),
            )

        wandb.log({"generations": generation_table})

        # Finish run
        wandb.finish()

        print("\n✓ Successfully logged to Weights & Biases")
        print(f"  Project: {args.wandb_project}")
        print(f"  Run: {run_name}")
        print("=" * 70)

    except Exception as e:
        print(f"\n⚠ Error logging to wandb: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Main function"""
    args: Namespace = parse_args()
    results: Dict[str, Any] = {}

    print("\n" + "=" * 60)
    print("FONTDIFFUSER SYNTHESIS DATA GENERATION MAGIC")
    print("=" * 60)

    try:
        # Load characters
        characters: List[str] = load_characters(
            args.characters, args.start_line, args.end_line
        )

        # Load style images with names
        style_paths_with_names: List[Tuple[str, str]] = load_style_images(
            args.style_images
        )

        print(f"\nInitializing font manager...")
        font_manager: FontManager = FontManager(args.ttf_path)

        print(f"\n📊 Configuration:")
        print(f"  Dataset split: {args.dataset_split}")
        print(
            f"  Characters: {len(characters)} (lines {args.start_line}-{args.end_line or 'end'})"
        )
        print(f"  Styles: {len(style_paths_with_names)}")
        print(f"  Output Directory: {args.output_dir}")
        print(f"  Checkpoint Directory: {args.ckpt_dir}")
        print(f"  Device: {args.device}")
        print(f"  Batch Size: {args.batch_size}")
        print(
            f"Will look for results checkpoint at {os.path.join(args.output_dir, 'results_checkpoint.json')}"
        )

        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize generation tracker
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        # Create args namespace for pipeline
        pipeline_args: Namespace = create_args_namespace(args)

        print("\nLoading FontDiffuser pipeline...")
        pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(pipeline_args)

        # Add this block to enable torch.compile if requested
        if getattr(args, "compile", False):
            import torch

            print("🔧 Compiling model components with torch.compile...")
            try:
                if hasattr(pipe.model, "unet"):
                    pipe.model.unet = torch.compile(pipe.model.unet)
                if hasattr(pipe.model, "style_encoder"):
                    pipe.model.style_encoder = torch.compile(pipe.model.style_encoder)
                if hasattr(pipe.model, "content_encoder"):
                    pipe.model.content_encoder = torch.compile(
                        pipe.model.content_encoder
                    )
                print("✓ Compilation complete.")
            except Exception as e:
                print(f"⚠ Compilation failed: {e}")

        evaluator: QualityEvaluator = QualityEvaluator(device=args.device)

        # Generate images
        results: Dict[str, Any] = batch_generate_images(
            pipe,
            characters,
            style_paths_with_names,
            args.output_dir,
            pipeline_args,
            evaluator,
            font_manager,
            generation_tracker,
        )

        # Evaluate if requested
        if args.evaluate and args.ground_truth_dir:
            results = evaluate_results(
                results, evaluator, args.ground_truth_dir, args.compute_fid
            )

        # Save final checkpoint
        print("\n💾 Saving final checkpoint...")
        save_checkpoint(results, args.output_dir)

        if args.use_wandb:
            log_to_wandb(results, args)

        print("\n" + "=" * 60)
        print("✅ GENERATION COMPLETE!")
        print("=" * 60)
        print(f"\nOutput structure:")
        print(f"  {args.output_dir}/")
        print(f"    ├── ContentImage/")
        print(f"    │   ├── U+XXXX_char_hash.png")
        print(f"    │   └── ...")
        print(f"    ├── TargetImage/")
        print(f"    │   ├── style0/")
        print(f"    │   │   ├── U+XXXX_char_style0_hash.png")
        print(f"    │   │   └── ...")
        print(f"    │   └── ...")
        print(f"    └── results_checkpoint.json ✅ (single source of truth)")

    except KeyboardInterrupt:
        print("\n\n⚠ Generation interrupted by user!")
        print("💾 Saving emergency checkpoint...")
        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
            print("✓ Latest state saved to results_checkpoint.json")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()

        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()
