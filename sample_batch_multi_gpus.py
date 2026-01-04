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
from argparse import ArgumentParser, Namespace

from filename_utils import compute_file_hash, get_content_filename, get_target_filename
from sample_optimized import (
    get_content_transform,
    get_style_transform,
    load_fontdiffuser_pipeline,
)
from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from utils import is_char_in_font, load_ttf, ttf2im
from utilities import get_hf_bar
from sample_batch import (
    FontManager,
    GenerationTracker,
    QualityEvaluator,
    parse_args,
    create_args_namespace,
    load_characters,
    load_style_images,
    save_checkpoint,
    sampling_batch_optimized,
    log_to_wandb,
)

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


def generate_content_images_with_accelerator(
    characters: List[str],
    font_manager: FontManager,
    output_dir: str,
    accelerator: Accelerator,
) -> Dict[str, str]:
    """Generate content images distributed across GPUs."""
    content_dir = os.path.join(output_dir, "ContentImage")

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
    chars_already_exist = []
    generated_new = 0
    
    with accelerator.split_between_processes(characters) as local_chars:
        for char in get_hf_bar(
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
                content_filename = get_content_filename(char)
                char_path: str = os.path.join(content_dir, content_filename)

                if os.path.exists(char_path):
                    logger.info(
                        f"  ✓ Content image already exists for '{char}' at {char_path}"
                    )
                    local_char_paths[char] = char_path
                    chars_already_exist.append(char)
                    continue

                font = font_manager.get_font(found_font)
                content_img: Image.Image = ttf2im(font=font, char=char)
                content_img.save(char_path)
                
                logger.info(
                    f"  ✓ Generated new content image for '{char}' at {char_path}."
                )
                local_char_paths[char] = char_path
                generated_new += 1

            except Exception as e:
                logger.warning(f"  ✗ Error generating '{char}': {e}")

    # Gather results from all GPUs
    accelerator.wait_for_everyone()
    all_char_paths_list = gather_object(local_char_paths)

    # Merge results on main process
    if accelerator.is_main_process:
        merged_char_paths = {}
        for paths in all_char_paths_list:
            merged_char_paths.update(paths)
        
        logger.info(f"{'=' * 60}")
        logger.info(f"Content Image Generation Summary:")
        logger.info(f"  Total characters:       {len(characters)}")
        logger.info(f"  Generated (new):        {generated_new}")
        logger.info(f"  Already exist (reused): {len(chars_already_exist)}")
        logger.info(f"  Total usable:           {len(merged_char_paths)}")
        logger.info("=" * 60)
        
        return merged_char_paths
    else:
        accelerator.wait_for_everyone()
        return {}
    
def batch_generate_images_with_accelerator(
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
    """Main batch generation with multi-GPU support."""

    # Generate content images
    char_paths = generate_content_images_with_accelerator(
        characters, font_manager, output_dir, accelerator
    )

    if accelerator.is_main_process and not char_paths:
        raise ValueError("No content images generated")

    # Initialize results
    all_chars_in_checkpoint = set(
        gen.get("character", "") for gen in generation_tracker.generations
    )
    all_styles_in_checkpoint = set(
        gen.get("style", "") for gen in generation_tracker.generations
    )
    all_chars_in_checkpoint.update(char_paths.keys())

    if accelerator.is_main_process:
        results = {
            "generations": generation_tracker.generations.copy(),
            "metrics": {"lpips": [], "ssim": [], "inference_times": []},
            "dataset_split": args.dataset_split,
            "fonts": font_manager.get_font_names(),
            "characters": sorted(list(all_chars_in_checkpoint)),
            "styles": sorted(list(all_styles_in_checkpoint)),
            "total_chars": len(all_chars_in_checkpoint),
            "total_styles": len(all_styles_in_checkpoint),
        }
    else:
        results = {}

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
        logger.info(
            f"Generating images: {len(characters)} chars × {len(style_paths_with_names)} styles"
        )
        logger.info(f"Using {accelerator.num_processes} GPUs")
        logger.info(f"Primary font: {primary_font}")

    # Counters
    generated_count = 0
    skipped_count = 0
    failed_count = 0

    # Distribute styles across GPUs
    with accelerator.split_between_processes(style_paths_with_names) as local_styles:
        for style_idx, (style_path, style_name) in enumerate(
            get_hf_bar(
                local_styles,
                desc=f"GPU {accelerator.process_index}",
                disable=not accelerator.is_local_main_process,
            )
        ):
            try:
                style_dir = os.path.join(target_base_dir, style_name)
                os.makedirs(style_dir, exist_ok=True)

                # Filter characters not yet generated
                chars_to_generate = [
                    char
                    for char in characters
                    if not generation_tracker.is_generated(
                        char, style_name, primary_font
                    )
                ]

                if not chars_to_generate:
                    skipped_count += len(characters)
                    continue

                # Generate batch
                images, valid_chars, batch_time = sampling_batch_optimized(
                    args,
                    pipe,
                    chars_to_generate,
                    style_path,
                    font_manager,
                    primary_font,
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

                        # Add generation record
                        if accelerator.is_main_process:
                            generation_record = {
                                "character": char,
                                "char_code": f"U+{ord(char):04X}",
                                "style": style_name,
                                "font": primary_font,
                                "content_image_path": content_path_rel,
                                "target_image_path": target_path_rel,
                                "content_hash": compute_file_hash(char, "", primary_font),
                                "target_hash": compute_file_hash(
                                    char, style_name, primary_font
                                ),
                                "content_filename": content_filename,
                                "target_filename": target_filename,
                            }
                            results["generations"].append(generation_record)
                            generation_tracker.add_generation(generation_record)
                            generated_count += 1

                    except Exception as e:
                        logger.warning(f"Error saving '{char}': {e}")
                        failed_count += 1

                # Record inference time
                if batch_time is not None:
                    results["metrics"]["inference_times"].append(
                        {
                            "style": style_name,
                            "font": primary_font,
                            "total_time": batch_time,
                            "num_images": len(images),
                            "time_per_image": batch_time / len(images),
                        }
                    )

                # Save checkpoint periodically
                if args.save_interval > 0 and (style_idx + 1) % args.save_interval == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(results, args.output_dir)
                        logger.info(
                            f"Checkpoint saved at {style_idx + 1}/{len(local_styles)} styles"
                        )

            except Exception as e:
                logger.error(f"Error processing {style_name}: {e}")
                failed_count += (
                    len(chars_to_generate)
                    if "chars_to_generate" in locals()
                    else len(characters)
                )

    # Final summary on main process
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Generated: {generated_count} images")
        logger.info(f"Skipped: {skipped_count} images")
        logger.info(f"Failed: {failed_count} images")
        logger.info(f"Total characters: {len(all_chars_in_checkpoint)}")
        logger.info(f"Total styles: {len(all_styles_in_checkpoint)}")
        logger.info("=" * 60)

        # Update results metadata
        results["characters"] = sorted(list(all_chars_in_checkpoint))
        results["styles"] = sorted(list(all_styles_in_checkpoint))
        results["total_chars"] = len(all_chars_in_checkpoint)
        results["total_styles"] = len(all_styles_in_checkpoint)

        save_checkpoint(results, args.output_dir)

    return results


def evaluate_results_with_accelerator(
    results: Dict[str, Any],
    evaluator: QualityEvaluator,
    output_dir: str,
    ground_truth_dir: Optional[str] = None,
    compute_fid: bool = False,
    accelerator: Optional[Accelerator] = None,
) -> Dict[str, Any]:
    """Evaluate generated images on main process."""

    if not accelerator or not accelerator.is_main_process:
        return results

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        logger.info("No ground truth directory provided, skipping evaluation")
        return results

    logger.info("=" * 60)
    logger.info("EVALUATING GENERATED IMAGES")
    logger.info("=" * 60)

    lpips_scores = []
    ssim_scores = []
    evaluated = 0

    target_base_dir = os.path.join(output_dir, "TargetImage")

    for gen in get_hf_bar(
        results["generations"],
        desc="Evaluating",
        disable=not accelerator.is_main_process,
    ):
        char = gen["character"]
        style = gen["style"]
        target_path = os.path.join(target_base_dir, style, gen["target_filename"])

        if not os.path.exists(target_path):
            continue

        # Try to find ground truth
        gt_filename = get_target_filename(char, style)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            continue

        try:
            generated_img = Image.open(target_path).convert("RGB")
            gt_img = Image.open(gt_path).convert("RGB")

            lpips_score = evaluator.compute_lpips(generated_img, gt_img)
            if lpips_score >= 0:
                lpips_scores.append(lpips_score)
                gen["lpips"] = lpips_score

            ssim_score = evaluator.compute_ssim(generated_img, gt_img)
            if ssim_score >= 0:
                ssim_scores.append(ssim_score)
                gen["ssim"] = ssim_score

            evaluated += 1
        except Exception as e:
            logger.warning(f"Error evaluating {char}/{style}: {e}")

    # Log metrics
    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }
        logger.info(f"LPIPS: mean={results['metrics']['lpips']['mean']:.4f}")

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }
        logger.info(f"SSIM: mean={results['metrics']['ssim']['mean']:.4f}")

    logger.info(f"Evaluated {evaluated} image pairs")
    logger.info("=" * 60)

    return results


def main():
    """Main entry point."""
    args = parse_args()
    results: Dict[str, Any] = {}

    logger.info("=" * 60)
    logger.info("FONTDIFFUSER SYNTHESIS DATA GENERATION MAGIC")
    logger.info("=" * 60)

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if args.fp16 else "no",
    )

    try:
        # Load characters
        characters: List[str] = load_characters(
            args.characters, args.start_line, args.end_line
        )

        # Load style images with names
        style_paths_with_names: List[Tuple[str, str]] = load_style_images(
            args.style_images
        )

        if accelerator.is_main_process:
            logger.info(f"Initializing font manager...")
        font_manager: FontManager = FontManager(args.ttf_path)
        if accelerator.is_main_process:
            logger.info(f"✓ Loaded {len(font_manager.get_font_names())} fonts.")

        if accelerator.is_main_process:
            logger.info(f"📊 Configuration:")
            logger.info(f"  Dataset split: {args.dataset_split}")
            logger.info(
                f"  Characters: {len(characters)} (lines {args.start_line}-{args.end_line or 'end'})"
            )
            logger.info(f"  Styles: {len(style_paths_with_names)}")
            logger.info(f"  Output Directory: {args.output_dir}")
            logger.info(f"  Checkpoint Directory: {args.ckpt_dir}")
            logger.info(f"  Device: {args.device}")
            logger.info(f"  Batch Size: {args.batch_size}")
            logger.info(f"  Using {accelerator.num_processes} GPUs")
            logger.info(
                f"  Results checkpoint path: {os.path.join(args.output_dir, 'results_checkpoint.json')}"
            )

        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize generation tracker
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        # Create args namespace for pipeline
        pipeline_args: Namespace = create_args_namespace(args)

        if accelerator.is_main_process:
            logger.info("\nLoading FontDiffuser pipeline...")
        pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(pipeline_args)
        pipe = accelerator.prepare(pipe)

        # Add this block to enable torch.compile if requested
        if getattr(args, "compile", False):
            import torch

            if accelerator.is_main_process:
                logger.info("🔧 Compiling model components with torch.compile...")
            try:
                if hasattr(pipe.model.config, "unet"):
                    pipe.model.config.unet = torch.compile(pipe.model.config.unet)
                if hasattr(pipe.model.config, "style_encoder"):
                    pipe.model.config.style_encoder = torch.compile(
                        pipe.model.config.style_encoder
                    )
                if hasattr(pipe.model.config, "content_encoder"):
                    pipe.model.config.content_encoder = torch.compile(
                        pipe.model.config.content_encoder
                    )
                if accelerator.is_main_process:
                    logger.info("✓ Compilation complete.")
            except Exception as e:
                if accelerator.is_main_process:
                    logger.info(f"⚠ Compilation failed: {e}")

        evaluator: QualityEvaluator = QualityEvaluator(device=args.device)

        # Generate images
        results: Dict[str, Any] = batch_generate_images_with_accelerator(
            pipe,
            characters,
            style_paths_with_names,
            args.output_dir,
            pipeline_args,
            evaluator,
            font_manager,
            generation_tracker,
            accelerator,
        )

        # Evaluate if requested
        if args.evaluate and args.ground_truth_dir and accelerator.is_main_process:
            results = evaluate_results_with_accelerator(
                results,
                evaluator,
                args.output_dir,
                args.ground_truth_dir,
                args.compute_fid,
                accelerator,
            )

        # Save final checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            logger.info("\n💾 Saving final checkpoint...")
            save_checkpoint(results, args.output_dir)

            if args.use_wandb:
                log_to_wandb(results, args)

        accelerator.wait_for_everyone()  # ← Add this
        
        if accelerator.is_main_process:
            logger.info("=" * 60)
            logger.info("✅ GENERATION COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"Output structure:")
            logger.info(f"  {args.output_dir}/")
            logger.info(f"    ├── ContentImage/")
            logger.info(f"    │   ├── U+XXXX_char_hash.png")
            logger.info(f"    │   └── ...")
            logger.info(f"    ├── TargetImage/")
            logger.info(f"    │   ├── style0/")
            logger.info(f"    │   │   ├── U+XXXX_char_style0_hash.png")
            logger.info(f"    │   │   └── ...")
            logger.info(f"    │   └── ...")
            logger.info(f"    └── results_checkpoint.json ✅ (single source of truth)")

        # accelerator.wait_for_everyone()
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Generation interrupted by user!")
        logger.info("💾 Saving emergency checkpoint...")
        if "results" in locals() and results:
            if accelerator.is_main_process:
                save_checkpoint(results, args.output_dir)
                logger.info("✓ Latest state saved to results_checkpoint.json")
        sys.exit(1)

    except Exception as e:
        logger.error(f"✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()

        if "results" in locals() and results:
            if accelerator.is_main_process:
                save_checkpoint(results, args.output_dir)
        sys.exit(1)

    finally:
        try:
            accelerator.free_memory()
            # Only destroy if distributed
            if accelerator.num_processes > 1:
                import torch.distributed
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
            logger.info("Cleanup completed successfully")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

if __name__ == "__main__":
    main()
