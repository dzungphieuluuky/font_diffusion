"""
Multi-GPU batch sampling and evaluation for FontDiffuser using Accelerate.

Uses hash-based file naming, results_checkpoint.json as single source of truth,
and supports resumable generation with proper multi-GPU distribution.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm.auto import tqdm

from filename_utils import compute_file_hash, get_content_filename, get_target_filename
from sample_optimized import (
    get_content_transform,
    get_style_transform,
    load_fontdiffuser_pipeline,
)
from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from utils import is_char_in_font, load_ttf, ttf2im

# Configure logging for multi-GPU
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    logger.warning("lpips not available. Install with: pip install lpips")

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    FID_AVAILABLE = False
    logger.warning("pytorch-fid not available. Install with: pip install pytorch-fid")

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False
    logger.warning("scikit-image not available. Install with: pip install scikit-image")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class FontManager:
    """Manages multiple font files."""
    
    def __init__(self, ttf_path: str):
        """Initialize font manager.
        
        Args:
            ttf_path: Path to font file, directory, or glob pattern
        """
        self.fonts: Dict[str, Dict[str, Any]] = {}
        self.font_paths: List[str] = []
        self._load_fonts(ttf_path)
        
    def _load_fonts(self, ttf_path: str) -> None:
        """Load fonts from path."""
        if "*" in ttf_path:
            import glob
            font_files = sorted(glob.glob(ttf_path))
            if not font_files:
                raise ValueError(f"No fonts found for pattern: {ttf_path}")
            self._load_font_list(font_files)
            
        elif os.path.isfile(ttf_path):
            font_name = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": load_ttf(ttf_path),
                "name": font_name,
            }
            self.font_paths = [ttf_path]
            logger.info(f"Loaded font: {font_name}")
            
        elif os.path.isdir(ttf_path):
            font_exts = {".ttf", ".otf", ".TTF", ".OTF"}
            font_files = sorted([
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_exts
            ])
            if not font_files:
                raise ValueError(f"No fonts found in directory: {ttf_path}")
            self._load_font_list(font_files)
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")
            
    def _load_font_list(self, font_files: List[str]) -> None:
        """Load multiple font files."""
        logger.info(f"Loading {len(font_files)} fonts...")
        for font_path in font_files:
            font_name = os.path.splitext(os.path.basename(font_path))[0]
            try:
                self.fonts[font_name] = {
                    "path": font_path,
                    "font": load_ttf(font_path),
                    "name": font_name,
                }
                logger.info(f"  ✓ {font_name}")
            except Exception as e:
                logger.warning(f"  ✗ Failed to load {font_name}: {e}")
        self.font_paths = font_files
        logger.info(f"Successfully loaded {len(self.fonts)} fonts")
        
    def get_font_names(self) -> List[str]:
        """Get list of loaded font names."""
        return list(self.fonts.keys())
        
    def get_font(self, font_name: str):
        """Get font object by name."""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["font"]
        
    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name."""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["path"]
        
    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font."""
        font_path = self.get_font_path(font_name)
        return is_char_in_font(font_path, char)
        
    def get_available_chars_for_font(
        self, font_name: str, characters: List[str]
    ) -> List[str]:
        """Get list of characters available in specific font."""
        return [char for char in characters if self.is_char_in_font(font_name, char)]


class GenerationTracker:
    """Tracks generated (character, style, font) combinations using hashes."""
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize tracker.
        
        Args:
            checkpoint_path: Path to results_checkpoint.json
        """
        self.generated_hashes: Set[str] = set()
        self.generations: List[Dict[str, Any]] = []
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint."""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                
            raw_generations = results.get("generations", [])
            seen_hashes = set()
            unique_generations = []
            duplicate_count = 0
            
            for gen in raw_generations:
                target_hash = gen.get("target_hash")
                if not target_hash:
                    char = gen.get("character", "")
                    style = gen.get("style", "")
                    font = gen.get("font", "")
                    if not char or not style:
                        continue
                    target_hash = compute_file_hash(char, style, font)
                    
                if target_hash in seen_hashes:
                    duplicate_count += 1
                    continue
                    
                seen_hashes.add(target_hash)
                self.generated_hashes.add(target_hash)
                unique_generations.append(gen)
                
            self.generations = unique_generations
            logger.info(f"Loaded checkpoint: {len(self.generations)} unique generations")
            if duplicate_count > 0:
                logger.info(f"  Removed {duplicate_count} duplicates")
                
        except Exception as e:
            logger.warning(f"Error loading checkpoint: {e}")
            
    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if combination has been generated."""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes
        
    def mark_generated(self, char: str, style: str, font: str = "") -> None:
        """Mark combination as generated."""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)
        
    def add_generation(self, generation: Dict[str, Any]) -> None:
        """Add generation record."""
        self.generations.append(generation)
        char = generation.get("character", "")
        style = generation.get("style", "")
        font = generation.get("font", "")
        self.mark_generated(char, style, font)


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID."""
    
    def __init__(self, device: str = "cuda"):
        """Initialize evaluator.
        
        Args:
            device: Device for computation
        """
        self.device = device
        
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
            
        self.transform = transforms.ToTensor()
        
    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images."""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0
            
        try:
            img1_t = self.transform(img1).unsqueeze(0).to(self.device) * 2 - 1
            img2_t = self.transform(img2).unsqueeze(0).to(self.device) * 2 - 1
            
            with torch.inference_mode():
                score = self.lpips_fn(img1_t, img2_t).item()
            return score
        except Exception as e:
            logger.warning(f"LPIPS computation failed: {e}")
            return -1.0
            
    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images."""
        if not SSIM_AVAILABLE:
            return -1.0
            
        try:
            img1_gray = np.array(img1.convert("L"))
            img2_gray = np.array(img2.convert("L"))
            score = ssim(img1_gray, img2_gray, data_range=255)
            return score
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            return -1.0
            
    def save_image(self, image: Image.Image, path: str) -> None:
        """Save PIL image to path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch sampling for FontDiffuser"
    )
    
    # Input/Output
    parser.add_argument("--characters", type=str, required=True,
                       help="Comma-separated characters or path to text file")
    parser.add_argument("--start_line", type=int, default=1,
                       help="Start line number (1-indexed)")
    parser.add_argument("--end_line", type=int, default=None,
                       help="End line number (inclusive)")
    parser.add_argument("--style_images", type=str, required=True,
                       help="Comma-separated paths or directory")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--ground_truth_dir", type=str, default=None,
                       help="Ground truth directory for evaluation")
    
    # Model
    parser.add_argument("--ckpt_dir", type=str, required=True,
                       help="Checkpoint directory")
    parser.add_argument("--ttf_path", type=str, required=True,
                       help="Path to font file or directory")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    # Generation
    parser.add_argument("--num_inference_steps", type=int, default=15,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size per GPU")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # Image sizes
    parser.add_argument("--style_image_size", type=int, default=96,
                       help="Style image size")
    parser.add_argument("--content_image_size", type=int, default=96,
                       help="Content image size")
    
    # Optimization
    parser.add_argument("--fp16", action="store_true",
                       help="Use FP16 precision")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=10,
                       help="Save every N styles")
    
    # Evaluation
    parser.add_argument("--evaluate", action="store_true", default=True,
                       help="Evaluate generated images")
    parser.add_argument("--compute_fid", action="store_true",
                       help="Compute FID")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true",
                       help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="fontdiffuser-eval",
                       help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Wandb run name")
    
    parser.add_argument("--dataset_split", type=str, default="train",
                       help="Dataset split name")
    
    # DPM-Solver parameters
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--algorithm_type", type=str, default="dpmsolver++")
    parser.add_argument("--skip_type", type=str, default="time_uniform")
    parser.add_argument("--method", type=str, default="multistep")
    parser.add_argument("--t_start", type=float, default=1.0)
    parser.add_argument("--t_end", type=float, default=1e-3)
    parser.add_argument("--content_encoder_downsample_size", type=int, default=3)
    
    return parser.parse_args()


def load_characters(
    characters_arg: str,
    start_line: int = 1,
    end_line: Optional[int] = None
) -> List[str]:
    """Load characters from file or comma-separated string."""
    chars = []
    
    if os.path.isfile(characters_arg):
        with open(characters_arg, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            
        start_idx = max(0, start_line - 1)
        end_idx = len(all_lines) if end_line is None else min(len(all_lines), end_line)
        
        if start_idx >= len(all_lines):
            raise ValueError(
                f"start_line ({start_line}) exceeds file length ({len(all_lines)})"
            )
        if start_idx >= end_idx:
            raise ValueError(f"Invalid line range: {start_line} to {end_line}")
            
        logger.info(f"Loading characters from {characters_arg}")
        logger.info(f"  Lines {start_line}-{end_idx} ({end_idx - start_idx} lines)")
        
        for line_num, line in enumerate(all_lines[start_idx:end_idx], start=start_line):
            char = line.strip()
            if not char:
                continue
            if len(char) != 1:
                logger.warning(f"Skipping line {line_num}: invalid character '{char}'")
                continue
            chars.append(char)
    else:
        for c in [x.strip() for x in characters_arg.split(",") if x.strip()]:
            if len(c) != 1:
                raise ValueError(f"Invalid character: '{c}'")
            chars.append(c)
            
    if not chars:
        raise ValueError("No valid characters loaded")
        
    logger.info(f"Loaded {len(chars)} characters")
    return chars


def load_style_images(style_images_arg: str) -> List[Tuple[str, str]]:
    """Load style image paths and extract style names.
    
    Returns:
        List of (path, style_name) tuples
    """
    if os.path.isdir(style_images_arg):
        image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        style_paths = sorted([
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ])
        logger.info(f"Loaded {len(style_paths)} style images from directory")
        return [(p, os.path.splitext(os.path.basename(p))[0]) for p in style_paths]
    else:
        style_paths = [p.strip() for p in style_images_arg.split(",")]
        return [(p, os.path.splitext(os.path.basename(p))[0]) for p in style_paths]


def create_args_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Create args namespace for FontDiffuser pipeline."""
    try:
        from configs.fontdiffuser import get_parser
        parser = get_parser()
        default_args = parser.parse_args([])
    except Exception:
        default_args = argparse.Namespace()
        
    # Copy all attributes
    for key, value in vars(args).items():
        setattr(default_args, key, value)
        
    # Convert image sizes to tuples
    if isinstance(default_args.style_image_size, int):
        default_args.style_image_size = (
            default_args.style_image_size,
            default_args.style_image_size
        )
    if isinstance(default_args.content_image_size, int):
        default_args.content_image_size = (
            default_args.content_image_size,
            default_args.content_image_size
        )
        
    # Set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    default_args.resolution = 96
    default_args.guidance_type = "classifier-free"
    default_args.model_type = "noise"
    default_args.correcting_x0_fn = None
    
    return default_args


def save_checkpoint(results: Dict[str, Any], output_dir: str) -> None:
    """Save results checkpoint."""
    checkpoint_path = os.path.join(output_dir, "results_checkpoint.json")
    
    if "metrics" not in results:
        results["metrics"] = {"lpips": [], "ssim": [], "inference_times": []}
        
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Saved checkpoint: {len(results.get('generations', []))} generations")


def generate_content_images(
    characters: List[str],
    font_manager: FontManager,
    output_dir: str,
    accelerator: Accelerator,
) -> Dict[str, str]:
    """Generate content images distributed across GPUs.
    
    Args:
        characters: List of characters to generate
        font_manager: Font manager instance
        output_dir: Output directory
        accelerator: Accelerator instance
        
    Returns:
        Dictionary mapping character to image path
    """
    content_dir = os.path.join(output_dir, "ContentImage")
    
    # Main process creates directory
    if accelerator.is_main_process:
        os.makedirs(content_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")
        
    if accelerator.is_main_process:
        logger.info(f"Generating content images for {len(characters)} characters")
        
    # Split characters across GPUs
    local_char_paths = {}
    with accelerator.split_between_processes(characters) as local_chars:
        for char in tqdm(
            local_chars,
            desc=f"GPU {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        ):
            # Find font containing character
            found_font = None
            for font_name in font_names:
                if font_manager.is_char_in_font(font_name, char):
                    found_font = font_name
                    break
                    
            if not found_font:
                continue
                
            try:
                # Generate content image
                font = font_manager.get_font(found_font)
                content_img = ttf2im(font=font, char=char)
                content_filename = get_content_filename(char)
                char_path = os.path.join(content_dir, content_filename)
                
                # Skip if already exists
                if not os.path.exists(char_path):
                    content_img.save(char_path)
                    
                local_char_paths[char] = char_path
            except Exception as e:
                logger.warning(f"Error generating '{char}': {e}")
                
    # Gather results from all GPUs
    accelerator.wait_for_everyone()
    all_char_paths_list = gather_object([local_char_paths])
    
    # Merge results on main process
    if accelerator.is_main_process:
        merged_char_paths = {}
        for paths in all_char_paths_list:
            merged_char_paths.update(paths)
        logger.info(f"Generated {len(merged_char_paths)} content images")
        return merged_char_paths
    else:
        return {}


def sampling_batch(
    args: argparse.Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_image_path: Union[str, Image.Image],
    font_manager: FontManager,
    font_name: str,
) -> Tuple[Optional[List[Image.Image]], Optional[List[str]], Optional[float]]:
    """Batch sampling for multiple characters.
    
    Args:
        args: Arguments
        pipe: Pipeline
        characters: List of characters
        style_image_path: Style image path or PIL image
        font_manager: Font manager
        font_name: Font name to use
        
    Returns:
        Tuple of (images, valid_chars, batch_time)
    """
    # Get available characters for this font
    available_chars = font_manager.get_available_chars_for_font(font_name, characters)
    if not available_chars:
        return None, None, None
        
    try:
        # Load style image
        if isinstance(style_image_path, str):
            style_image = Image.open(style_image_path).convert("RGB")
        else:
            style_image = style_image_path.convert("RGB")
            
        style_transform = get_style_transform(args.style_image_size)
        font = font_manager.get_font(font_name)
        content_transform = get_content_transform(args.content_image_size)
        
        # Generate content images
        content_images = []
        for char in available_chars:
            try:
                content_image = ttf2im(font=font, char=char)
                content_images.append(content_transform(content_image))
            except Exception as e:
                logger.warning(f"Error processing '{char}': {e}")
                
        if not content_images:
            return None, None, None
            
        # Prepare batches
        content_batch = torch.stack(content_images)
        style_batch = style_transform(style_image)[None, :].repeat(
            len(content_images), 1, 1, 1
        )
        
        with torch.inference_mode():
            dtype = torch.float16 if args.fp16 else torch.float32
            content_batch = content_batch.to(args.device, dtype=dtype)
            style_batch = style_batch.to(args.device, dtype=dtype)
            
            start = time.perf_counter()
            
            # Process in batches
            all_images = []
            for i in range(0, len(content_batch), args.batch_size):
                batch_content = content_batch[i:i + args.batch_size]
                batch_style = style_batch[i:i + args.batch_size]
                
                images = pipe.generate(
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
                
            end = time.perf_counter()
            total_time = end - start
            
            return all_images, available_chars, total_time
            
    except Exception as e:
        logger.error(f"Batch sampling failed: {e}")
        return None, None, None


def batch_generate_images(
    pipe: FontDiffuserDPMPipeline,
    characters: List[str],
    style_paths_with_names: List[Tuple[str, str]],
    output_dir: str,
    args: argparse.Namespace,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    """Main batch generation with multi-GPU support.
    
    Args:
        pipe: Pipeline
        characters: List of characters
        style_paths_with_names: List of (path, name) tuples
        output_dir: Output directory
        args: Arguments
        evaluator: Evaluator
        font_manager: Font manager
        generation_tracker: Generation tracker
        accelerator: Accelerator instance
        
    Returns:
        Results dictionary
    """
    # Generate content images
    char_paths = generate_content_images(
        characters, font_manager, output_dir, accelerator
    )
    
    if accelerator.is_main_process and not char_paths:
        raise ValueError("No content images generated")
        
    # Initialize results
    all_chars = set(gen.get("character", "") for gen in generation_tracker.generations)
    all_styles = set(gen.get("style", "") for gen in generation_tracker.generations)
    all_chars.update(char_paths.keys())
    
    results = {
        "generations": generation_tracker.generations.copy() if accelerator.is_main_process else [],
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": args.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": sorted(all_chars),
        "styles": sorted(all_styles),
        "total_chars": len(all_chars),
        "total_styles": len(all_styles),
    }
    
    # Setup directories
    target_base_dir = os.path.join(output_dir, "TargetImage")
    if accelerator.is_main_process:
        os.makedirs(target_base_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    
    # Get primary font
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")
    primary_font = font_names[0]
    
    if accelerator.is_main_process:
        logger.info(f"Generating images: {len(characters)} chars × {len(style_paths_with_names)} styles")
        logger.info(f"Using {accelerator.num_processes} GPUs")
        logger.info(f"Primary font: {primary_font}")
        
    # Counters
    generated_count = 0
    skipped_count = 0
    
    # Distribute styles across GPUs
    with accelerator.split_between_processes(style_paths_with_names) as local_styles:
        for style_idx, (style_path, style_name) in enumerate(tqdm(
            local_styles,
            desc=f"GPU {accelerator.process_index}",
            disable=not accelerator.is_local_main_process,
        )):
            style_dir = os.path.join(target_base_dir, style_name)
            os.makedirs(style_dir, exist_ok=True)
            
            # Filter characters not yet generated
            chars_to_generate = [
                char for char in characters
                if not generation_tracker.is_generated(char, style_name, primary_font)
            ]
            
            if not chars_to_generate:
                skipped_count += len(characters)
                continue
                
            # Generate batch
            images, valid_chars, batch_time = sampling_batch(
                args, pipe, chars_to_generate, style_path, font_manager, primary_font
            )
            
            if images is None:
                skipped_count += len(chars_to_generate)
                continue
                
            # Save images (each GPU saves its own)
            for char, img in zip(valid_chars, images):
                try:
                    target_filename = get_target_filename(char, style_name)
                    img_path = os.path.join(style_dir, target_filename)
                    
                    content_filename = get_content_filename(char)
                    content_path_rel = f"ContentImage/{content_filename}"
                    target_path_rel = f"TargetImage/{style_name}/{target_filename}"
                    
                    evaluator.save_image(img, img_path)
                    
                    # Create generation record
                    generation_record = {
                        "character": char,
                        "style": style_name,
                        "font": primary_font,
                        "content_image_path": content_path_rel,
                        "target_image_path": target_path_rel,
                        "content_hash": compute_file_hash(char, "", primary_font),
                        "target_hash": compute_file_hash(char, style_name, primary_font),
                    }
                    
                    # Add to local results
                    results["generations"].append(generation_record)
                    generation_tracker.add_generation(generation_record)
                    
                    all_chars.add(char)
                    all_styles.add(style_name)
                    generated_count += 1
                    
                except Exception as e:
                    logger.error(f"Error saving '{char}': {e}")
                    
            # Update results metadata
            results["characters"] = sorted(all_chars)
            results["styles"] = sorted(all_styles)
            results["total_chars"] = len(all_chars)
            results["total_styles"] = len(all_styles)
            
            # Track timing
            if batch_time:
                results["metrics"]["inference_times"].append({
                    "style": style_name,
                    "font": primary_font,
                    "total_time": batch_time,
                    "num_images": len(images),
                    "time_per_image": batch_time / len(images),
                })
                
            # Periodic checkpoint (main process only)
            if accelerator.is_main_process and args.save_interval > 0:
                if (style_idx + 1) % args.save_interval == 0:
                    save_checkpoint(results, output_dir)
                    
    # Synchronize and gather results
    accelerator.wait_for_everyone()
    
    if accelerator.is_main_process:
        # Gather all generation records from all GPUs
        all_results = gather_object([results])
        
        # Merge results
        merged_generations = []
        seen_hashes = set()
        for r in all_results:
            for gen in r.get("generations", []):
                target_hash = gen.get("target_hash")
                if target_hash not in seen_hashes:
                    seen_hashes.add(target_hash)
                    merged_generations.append(gen)
                    
        results["generations"] = merged_generations
        
        # Update character and style lists
        all_chars = set(g["character"] for g in merged_generations)
        all_styles = set(g["style"] for g in merged_generations)
        results["characters"] = sorted(all_chars)
        results["styles"] = sorted(all_styles)
        results["total_chars"] = len(all_chars)
        results["total_styles"] = len(all_styles)
        
        logger.info(f"Generation complete: {len(merged_generations)} unique pairs")
        logger.info(f"  Characters: {len(all_chars)}")
        logger.info(f"  Styles: {len(all_styles)}")
        
    return results


def main():
    """Main entry point with multi-GPU support."""
    # Initialize Accelerator
    accelerator = Accelerator()
    args = parse_args()
    
    if accelerator.is_main_process:
        logger.info(f"Starting generation with {accelerator.num_processes} GPUs")
        logger.info(f"Output directory: {args.output_dir}")
        
    try:
        # Load data
        characters = load_characters(args.characters, args.start_line, args.end_line)
        style_paths_with_names = load_style_images(args.style_images)
        font_manager = FontManager(args.ttf_path)
        
        # Create output directory
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        
        # Initialize tracker
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )
        
        # Create pipeline args
        pipeline_args = create_args_namespace(args)
        
        # Load pipeline
        if accelerator.is_main_process:
            logger.info("Loading FontDiffuser pipeline...")
        pipe = load_fontdiffuser_pipeline(pipeline_args)
        
        # Prepare pipeline with Accelerator
        pipe = accelerator.prepare(pipe)
        
        # Optional: torch.compile
        if args.compile and accelerator.is_main_process:
            logger.info("Compiling model...")
            try:
                pipe.model.unet = torch.compile(pipe.model.unet)
                pipe.model.style_encoder = torch.compile(pipe.model.style_encoder)
                pipe.model.content_encoder = torch.compile(pipe.model.content_encoder)
                logger.info("Compilation complete")
            except Exception as e:
                logger.warning(f"Compilation failed: {e}")
                
        # Create evaluator
        evaluator = QualityEvaluator(device=args.device)
        
        # Generate images
        results = batch_generate_images(
            pipe=pipe,
            characters=characters,
            style_paths_with_names=style_paths_with_names,
            output_dir=args.output_dir,
            args=pipeline_args,
            evaluator=evaluator,
            font_manager=font_manager,
            generation_tracker=generation_tracker,
            accelerator=accelerator,
        )
        
        # Save final checkpoint
        if accelerator.is_main_process:
            logger.info("Saving final checkpoint...")
            save_checkpoint(results, args.output_dir)
            
            if args.use_wandb and WANDB_AVAILABLE:
                logger.info("Logging to Wandb...")
                # Add wandb logging here if needed
                
            logger.info("Generation complete!")
            logger.info(f"  Total generations: {len(results['generations'])}")
            logger.info(f"  Output: {args.output_dir}")
            
    except KeyboardInterrupt:
        if accelerator.is_main_process:
            logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        if accelerator.is_main_process:
            logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()