"""
FontDiffuser Batch Processing with Robust Font Rendering
Fixed the cropping bug in ttf2im and added safety checks
"""

import os
import sys
import json
import time
import warnings
import copy
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import ast
import torch
import numpy as np
import cv2
from PIL import Image
import pygame
import pygame.freetype
from fontTools.ttLib import TTFont
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Suppress warnings
warnings.filterwarnings("ignore")

# Import FontDiffuser components
try:
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder,
    )
    from utils import (
        save_args_to_yaml,
        save_single_image,
        save_image_with_content_style,
    )

    from font_manager import FontManager, FontRenderer
except ImportError as e:
    print(f"Error importing FontDiffuser modules: {e}")
    print("Please ensure the required modules are in your Python path")
    sys.exit(1)


class FontDiffuserBatchProcessor:
    """Process Excel file and generate fonts for similar characters with robust rendering"""

    def __init__(self, args, pipe, font_manager: FontManager):
        self.args = args
        self.pipe = pipe
        self.font_manager = font_manager
        self.device = args.device

        # Create transforms
        self.content_transforms, self.style_transforms = self._create_transforms()

        # Statistics tracking
        self.stats = {
            "characters_processed": 0,
            "characters_skipped_no_font": 0,
            "characters_skipped_render_failed": 0,
            "generation_errors": 0,
            "fonts_used": defaultdict(int),
            "edge_cases_fixed": 0,
        }

    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create image transforms for content and style images"""
        content_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.args.content_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        style_transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.args.style_image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        return content_transforms, style_transforms

    def prepare_images(
        self, char: str, style_image_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Image.Image]]:
        """
        Prepare content and style images for a character with robust error handling
        """
        try:
            # Load style image
            if not os.path.exists(style_image_path):
                raise FileNotFoundError(f"Style image not found: {style_image_path}")

            style_image = Image.open(style_image_path).convert("RGB")
            style_image_tensor = self.style_transforms(style_image)[None, :].to(
                self.device
            )

            # Prepare content image using font manager
            if self.args.character_input:
                # Check font support
                if not self.font_manager.can_render_character(char):
                    print(
                        f"    Character '{char}' (U+{ord(char):04X}) not supported by any font"
                    )
                    self.stats["characters_skipped_no_font"] += 1
                    return None, None, None

                # Render character
                content_image = self.font_manager.render_character(char)
                if content_image is None:
                    print(f"    Failed to render character '{char}'")
                    self.stats["characters_skipped_render_failed"] += 1
                    return None, None, None

                # Check rendering quality
                img_array = np.array(content_image.convert("L"))

                # Detect potential rendering issues
                issues = self._detect_rendering_issues(img_array, char)

                if issues:
                    print(f"    Rendering issues for '{char}': {', '.join(issues)}")

                    # Try to fix common issues
                    if "edge_touching" in issues:
                        content_image = self._fix_edge_touching(content_image)
                        self.stats["edge_cases_fixed"] += 1

                content_image_pil = content_image.copy()

                # Track which font was used
                font_info = self.font_manager.get_supporting_font(char)
                if font_info:
                    self.stats["fonts_used"][font_info["name"]] += 1
            else:
                # If not using character input, use white image
                content_image = Image.new("RGB", (256, 256), color="white")
                content_image_pil = None

            # Convert content image to tensor
            content_image_tensor = self.content_transforms(content_image)[None, :].to(
                self.device
            )

            # Debug: Save the prepared content image
            if hasattr(self.args, "debug") and self.args.debug:
                debug_dir = Path(self.args.output_base_dir) / "debug" / "content_images"
                debug_dir.mkdir(parents=True, exist_ok=True)
                content_image.save(debug_dir / f"content_{char}.png")

            return content_image_tensor, style_image_tensor, content_image_pil

        except Exception as e:
            print(f"    Error preparing images for '{char}': {e}")
            self.stats["generation_errors"] += 1
            return None, None, None

    def _detect_rendering_issues(self, img_array: np.ndarray, char: str) -> List[str]:
        """Detect potential rendering issues in character image"""
        issues = []

        # Check if image is mostly one color
        if np.std(img_array) < 10:
            issues.append("low_contrast")

        # Check if character touches edges
        edge_thickness = 3
        edges = [
            img_array[:edge_thickness, :],  # Top
            img_array[-edge_thickness:, :],  # Bottom
            img_array[:, :edge_thickness],  # Left
            img_array[:, -edge_thickness:],  # Right
        ]

        edge_threshold = 200
        for i, edge in enumerate(edges):
            if np.any(edge < edge_threshold):
                issues.append("edge_touching")
                break

        # Check if character is too small
        binary = img_array < 240  # Threshold for character pixels
        character_pixels = np.sum(binary)
        total_pixels = img_array.size

        if character_pixels < total_pixels * 0.01:  # Less than 1% of pixels
            issues.append("too_small")

        return issues

    def _fix_edge_touching(self, image: Image.Image) -> Image.Image:
        """Fix edge-touching by adding padding"""
        # Convert to numpy
        img_array = np.array(image.convert("RGB"))

        # Add white border
        border_size = 20
        h, w = img_array.shape[:2]
        new_h, new_w = h + border_size * 2, w + border_size * 2

        new_array = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        new_array[border_size : border_size + h, border_size : border_size + w] = (
            img_array
        )

        return Image.fromarray(new_array).resize((h, w), Image.LANCZOS)

    def generate_character(
        self, char: str, style_image_path: str
    ) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
        """Generate a single character using FontDiffuser"""
        # Set seed for reproducibility if specified
        if self.args.seed:
            set_seed(seed=self.args.seed)

        # Prepare images
        content_tensor, style_tensor, content_pil = self.prepare_images(
            char, style_image_path
        )

        if content_tensor is None or style_tensor is None:
            return None, None

        try:
            # Generate using FontDiffuser pipeline
            with torch.no_grad():
                images = self.pipe.generate(
                    content_images=content_tensor,
                    style_images=style_tensor,
                    batch_size=1,
                    order=self.args.order,
                    num_inference_step=self.args.num_inference_steps,
                    content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                    t_start=self.args.t_start,
                    t_end=self.args.t_end,
                    dm_size=self.args.content_image_size,
                    algorithm_type=self.args.algorithm_type,
                    skip_type=self.args.skip_type,
                    method=self.args.method,
                    correcting_x0_fn=self.args.correcting_x0_fn,
                )

            self.stats["characters_processed"] += 1
            return images[0], content_pil

        except Exception as e:
            print(f"    Error generating character '{char}': {e}")
            self.stats["generation_errors"] += 1
            return None, None

    def process_excel_file(
        self,
        excel_path: str,
        base_output_dir: str,
        style_image_path: str = None,
        generate_input_char: bool = True,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
    ) -> Dict:
        """
        Process Excel file and generate fonts for similar characters

        Args:
            excel_path: Path to Excel file
            base_output_dir: Base directory for outputs
            style_image_path: Path to style image
            generate_input_char: Whether to generate the input character itself
            start_line: Starting row number (1-indexed, inclusive)
            end_line: Ending row number (1-indexed, inclusive)

        Returns:
            Dictionary with processing results and statistics
        """
        # Use provided style image or default from args
        if style_image_path is None:
            style_image_path = self.args.style_image_path

        # Create base output directory
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)

        # Save font statistics
        font_stats = self.font_manager.get_font_statistics()
        with open(
            Path(base_output_dir) / "font_statistics.json", "w", encoding="utf-8"
        ) as f:
            json.dump(font_stats, f, indent=2, ensure_ascii=False)

        print(f"\n{'=' * 70}")
        print(f"FONT STATISTICS")
        print(f"{'=' * 70}")
        print(f"Loaded {font_stats['total_fonts']} fonts:")
        for font_name, stats in font_stats["total_glyphs_by_font"].items():
            print(f"  • {font_name}: {stats} glyphs")
        print(f"{'=' * 70}\n")

        # Load Excel file
        print(f"Loading Excel file: {excel_path}")
        try:
            df = pd.read_excel(excel_path)
            total_rows = len(df)
            print(f"Excel file loaded with {total_rows} total rows")
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")

        # Check required columns
        required_columns = ["Input Character", "Top 20 Similar Characters"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Apply line range filters
        if start_line is not None or end_line is not None:
            # Convert to 0-indexed for pandas
            start_idx = (start_line - 1) if start_line else 0
            end_idx = (end_line - 1) if end_line else (total_rows - 1)

            # Validate range
            if start_idx < 0:
                print(f"Warning: start_line {start_line} is less than 1, using 1")
                start_idx = 0

            if end_idx >= total_rows:
                print(
                    f"Warning: end_line {end_line} exceeds total rows {total_rows}, using {total_rows}"
                )
                end_idx = total_rows - 1

            if start_idx > end_idx:
                print(
                    f"Warning: start_line ({start_line}) > end_line ({end_line}), swapping"
                )
                start_idx, end_idx = end_idx, start_idx

            # Slice the dataframe
            df = df.iloc[start_idx : end_idx + 1]
            print(
                f"Processing rows {start_line or 1} to {end_line or total_rows} "
                f"(indices {start_idx} to {end_idx}, {len(df)} rows)"
            )
        else:
            print(f"Processing all {total_rows} rows")

        # Process each row
        results_summary = {}

        for idx, row in df.iterrows():
            # Calculate actual row number in original Excel (1-indexed)
            original_row_num = (
                idx + 1 if start_line is None else start_idx + (idx - df.index[0]) + 1
            )

            input_char = str(row["Input Character"]).strip()
            similar_chars_str = row["Top 20 Similar Characters"]

            print(f"\n{'=' * 70}")
            print(
                f"Processing Row {original_row_num} (DataFrame index {idx}): "
                f"Input Character = '{input_char}'"
            )
            print(f"{'=' * 70}")

            # Parse similar characters
            similar_chars = self._parse_similar_characters(similar_chars_str)
            print(f"Found {len(similar_chars)} similar characters")

            # Check font support for all characters
            all_chars = (
                [input_char] + similar_chars if generate_input_char else similar_chars
            )
            font_mapping = self.font_manager.find_fonts_for_characters(all_chars)

            # Report font support
            unsupported = [c for c, f in font_mapping.items() if f is None]
            if unsupported:
                print(
                    f"  Characters without font support ({len(unsupported)}): "
                    f"{', '.join(unsupported[:5])}"
                    + ("..." if len(unsupported) > 5 else "")
                )

            # Create folder for this input character
            safe_char_name = self._sanitize_filename(input_char)
            char_output_dir = (
                Path(base_output_dir) / f"row_{original_row_num:04d}_{safe_char_name}"
            )
            char_output_dir.mkdir(exist_ok=True)

            # Save character information
            self._save_character_info(
                char_output_dir,
                input_char,
                similar_chars,
                font_mapping,
                style_image_path,
                original_row_num,
            )

            # Generate characters
            generated_chars = self._generate_characters_for_row(
                char_output_dir,
                input_char,
                similar_chars,
                style_image_path,
                generate_input_char,
                font_mapping,
                original_row_num,
            )

            # Update summary
            results_summary[input_char] = {
                "output_dir": str(char_output_dir),
                "excel_row": original_row_num,
                "generated_count": len(generated_chars),
                "similar_characters": similar_chars,
                "generated_chars": list(generated_chars.keys()),
                "font_mapping": {
                    k: v for k, v in font_mapping.items() if k in generated_chars
                },
            }

        # Save final summaries
        self._save_final_summaries(
            base_output_dir, excel_path, results_summary, start_line, end_line
        )

        return results_summary

    def _parse_similar_characters(self, similar_chars_str) -> List[str]:
        """Parse similar characters string from Excel"""
        try:
            if pd.isna(similar_chars_str):
                return []

            if isinstance(similar_chars_str, str):
                if similar_chars_str.startswith("["):
                    # Parse as Python list
                    return ast.literal_eval(similar_chars_str)[:20]
                else:
                    # Parse as comma-separated or other format
                    chars = (
                        str(similar_chars_str).strip("[]").replace("'", "").split(",")
                    )
                    return [c.strip() for c in chars if c.strip()][:20]
            else:
                return []

        except Exception as e:
            print(f"  Warning: Error parsing similar characters: {e}")
            return []

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize string to be safe for filenames"""
        # Keep Unicode characters but remove invalid path characters
        invalid_chars = r'<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")

        # Also remove control characters
        filename = "".join(c for c in filename if ord(c) >= 32)

        # Limit length
        if len(filename) > 50:
            filename = filename[:50]

        return filename

    def _save_character_info(
        self,
        output_dir: Path,
        input_char: str,
        similar_chars: List[str],
        font_mapping: Dict,
        style_image_path: str,
        excel_row: int,
    ):
        """Save character information to file"""
        info_path = output_dir / "character_info.json"
        info = {
            "input_character": input_char,
            "excel_row_number": excel_row,
            "similar_characters": similar_chars,
            "style_image": style_image_path,
            "font_mapping": font_mapping,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_characters": len(similar_chars) + 1,
        }

        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

    def _generate_characters_for_row(
        self,
        output_dir: Path,
        input_char: str,
        similar_chars: List[str],
        style_image_path: str,
        generate_input_char: bool,
        font_mapping: Dict,
        excel_row: int,
    ) -> Dict:
        """Generate all characters for a single row"""
        generated_chars = {}

        # Characters to generate
        chars_to_generate = []
        if generate_input_char:
            chars_to_generate.append(("input", input_char))

        for char in similar_chars:
            chars_to_generate.append(("similar", char))

        # Generate each character
        for char_type, char in chars_to_generate:
            print(
                f"  [{len(generated_chars) + 1}/{len(chars_to_generate)}] "
                f"Generating '{char}'...",
                end="",
            )

            # Check font support
            if font_mapping.get(char) is None:
                print(" ✗ (no font support)")
                continue

            # Generate character
            generated_image, content_pil = self.generate_character(
                char, style_image_path
            )

            if generated_image is not None:
                # Save the generated character
                save_success = self._save_generated_character(
                    output_dir,
                    char_type,
                    char,
                    generated_image,
                    content_pil,
                    style_image_path,
                )

                if save_success:
                    generated_chars[char] = {
                        "type": char_type,
                        "font_used": font_mapping[char],
                    }
                    print(" ✓")
                else:
                    print(" ✗ (save failed)")
            else:
                print(" ✗ (generation failed)")

        return generated_chars

    def _save_generated_character(
        self,
        output_dir: Path,
        char_type: str,
        char: str,
        generated_image: torch.Tensor,
        content_pil: Optional[Image.Image],
        style_image_path: str,
    ) -> bool:
        """Save generated character to appropriate location"""
        try:
            # Create directory structure
            if char_type == "input":
                char_dir = output_dir / "input_character"
            else:
                char_dir = (
                    output_dir / "similar_characters" / self._sanitize_filename(char)
                )

            char_dir.mkdir(parents=True, exist_ok=True)

            # Save single image
            save_single_image(save_dir=str(char_dir), image=generated_image)

            # Save with content and style if available
            if self.args.character_input and content_pil is not None:
                save_image_with_content_style(
                    save_dir=str(char_dir),
                    image=generated_image,
                    content_image_pil=content_pil,
                    content_image_path=None,
                    style_image_path=style_image_path,
                    resolution=self.args.resolution,
                )

            return True

        except Exception as e:
            print(f"    Error saving character '{char}': {e}")
            return False

    def _save_final_summaries(
        self,
        base_output_dir: str,
        excel_path: str,
        results_summary: Dict,
        start_line: Optional[int],
        end_line: Optional[int],
    ):
        """Save final summary files"""
        base_path = Path(base_output_dir)

        # Global summary
        global_summary = {
            "processing_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "excel_file": excel_path,
            "output_directory": base_output_dir,
            "line_range": {
                "start_line": start_line,
                "end_line": end_line,
                "processed_rows": len(results_summary),
            },
            "processing_statistics": self.stats,
            "detailed_results": results_summary,
        }

        # Save as JSON
        with open(base_path / "global_summary.json", "w", encoding="utf-8") as f:
            json.dump(global_summary, f, indent=2, ensure_ascii=False)

        # Also save as human-readable text
        with open(base_path / "global_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"FontDiffuser Batch Processing Summary\n")
            f.write(f"=" * 60 + "\n\n")

            f.write(f"Processing Completed: {global_summary['processing_completed']}\n")
            f.write(f"Excel File: {excel_path}\n")
            f.write(f"Output Directory: {base_output_dir}\n")

            if start_line or end_line:
                f.write(f"Line Range: {start_line or 'start'} to {end_line or 'end'}\n")
            f.write(f"Total Rows Processed: {len(results_summary)}\n\n")

            f.write(f"Processing Statistics:\n")
            f.write(f"  Characters Processed: {self.stats['characters_processed']}\n")
            f.write(
                f"  Characters Skipped (no font): {self.stats['characters_skipped_no_font']}\n"
            )
            f.write(
                f"  Characters Skipped (render failed): {self.stats['characters_skipped_render_failed']}\n"
            )
            f.write(f"  Generation Errors: {self.stats['generation_errors']}\n")
            f.write(f"  Edge Cases Fixed: {self.stats['edge_cases_fixed']}\n\n")

            f.write(f"Font Usage Statistics:\n")
            for font_name, count in self.stats["fonts_used"].items():
                f.write(f"  {font_name}: {count} characters\n")
            f.write("\n")

            f.write(f"Results by Input Character:\n")
            for input_char, info in results_summary.items():
                f.write(f"\n  {input_char} (Row {info['excel_row']}):\n")
                f.write(f"    Output: {info['output_dir']}\n")
                f.write(f"    Generated: {info['generated_count']} characters\n")

        print(f"\n{'=' * 70}")
        print(f"PROCESSING COMPLETE")
        print(f"{'=' * 70}")
        print(f"Statistics:")
        print(f"  • Characters processed: {self.stats['characters_processed']}")
        print(
            f"  • Characters skipped (no font): {self.stats['characters_skipped_no_font']}"
        )
        print(
            f"  • Characters skipped (render failed): {self.stats['characters_skipped_render_failed']}"
        )
        print(f"  • Generation errors: {self.stats['generation_errors']}")
        print(f"  • Edge cases fixed: {self.stats['edge_cases_fixed']}")
        print(f"\nFont usage:")
        for font_name, count in sorted(
            self.stats["fonts_used"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  • {font_name}: {count}")
        print(f"\nOutput saved to: {base_output_dir}")
        print(f"Global summary: {base_path / 'global_summary.json'}")


def parse_excel_batch_args():
    """Parse arguments for batch processing with line range support"""
    from configs.fontdiffuser import get_parser

    parser = get_parser()

    # Existing arguments
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Checkpoint directory"
    )
    parser.add_argument("--character_input", action="store_true", default=True)
    parser.add_argument(
        "--style_image_path",
        type=str,
        required=True,
        help="Path to style reference image",
    )
    parser.add_argument("--device", type=str, default="cuda:0")

    # Font arguments
    parser.add_argument(
        "--ttf_path",
        type=str,
        action="append",
        help="Path to TTF font file (can be specified multiple times)",
    )
    parser.add_argument(
        "--font_dir",
        type=str,
        default=None,
        help="Directory containing multiple TTF fonts (all .ttf files will be loaded)",
    )

    # Batch processing arguments
    parser.add_argument(
        "--excel_file",
        type=str,
        required=True,
        help="Path to Excel file with character data",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./fontdiffuser_batch_output",
        help="Base directory for all outputs",
    )
    parser.add_argument(
        "--skip_input_char",
        action="store_true",
        help="Skip generating the input character",
    )

    # NEW: Line range arguments
    parser.add_argument(
        "--start_line",
        type=int,
        default=None,
        help="Starting line number in Excel (1-indexed, inclusive)",
    )
    parser.add_argument(
        "--end_line",
        type=int,
        default=None,
        help="Ending line number in Excel (1-indexed, inclusive)",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (alternative to end_line)",
    )

    # Debug argument
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save debug images and additional information",
    )

    args = parser.parse_args()

    # Set image sizes
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    # Collect font paths
    font_paths = []

    # Add individual font paths
    if args.ttf_path:
        font_paths.extend([p for p in args.ttf_path if p])

    # Add fonts from directory
    if args.font_dir and os.path.exists(args.font_dir):
        font_paths.extend([str(p) for p in Path(args.font_dir).glob("*.ttf")])
        font_paths.extend([str(p) for p in Path(args.font_dir).glob("*.TTF")])

    # Remove duplicates
    font_paths = list(dict.fromkeys(font_paths))

    if not font_paths:
        # Try default location
        default_font = Path(__file__).parent / "fonts" / "default.ttf"
        if default_font.exists():
            font_paths = [str(default_font)]
        else:
            raise ValueError(
                "No font files specified or found. Use --ttf_path or --font_dir"
            )

    args.font_paths = font_paths

    # Validate line range arguments
    if args.start_line is not None and args.start_line < 1:
        print(f"Warning: start_line must be >= 1, got {args.start_line}")
        args.start_line = 1

    if args.end_line is not None and args.end_line < 1:
        print(f"Warning: end_line must be >= 1, got {args.end_line}")
        args.end_line = None

    if args.start_line is not None and args.end_line is not None:
        if args.start_line > args.end_line:
            print(
                f"Warning: start_line ({args.start_line}) > end_line ({args.end_line}), swapping"
            )
            args.start_line, args.end_line = args.end_line, args.start_line

    # Handle max_rows if specified (alternative to end_line)
    if args.max_rows is not None:
        if args.end_line is None and args.start_line is not None:
            args.end_line = args.start_line + args.max_rows - 1
        elif args.end_line is None:
            args.end_line = args.max_rows

    return args


def load_fontdiffuser_pipeline(args):
    """Load the FontDiffuser pipeline once"""
    print("Loading FontDiffuser model...")

    # Load model components
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))

    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))

    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))

    # Create model
    model = FontDiffuserModelDPM(
        unet=unet, style_encoder=style_encoder, content_encoder=content_encoder
    )
    model.to(args.device)

    # Load scheduler
    train_scheduler = build_ddpm_scheduler(args=args)

    # Create pipeline
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )

    print("✓ FontDiffuser pipeline loaded successfully")
    return pipe


def main_batch_processing():
    """Main function for batch processing Excel file with line range support"""
    args = parse_excel_batch_args()

    print(f"\n{'=' * 70}")
    print(f"FONTDIFFUSER BATCH PROCESSING")
    if args.start_line or args.end_line:
        print(f"Line Range: {args.start_line or 'start'} to {args.end_line or 'end'}")
    print(f"{'=' * 70}")

    # Create output directory
    output_dir = Path(args.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_args_to_yaml(args=args, output_file=str(output_dir / "batch_config.yaml"))

    # Initialize font manager with multiple fonts
    print(f"\nLoading fonts...")
    font_manager = FontManager(
        font_paths=args.font_paths, font_size=256, canvas_size=256
    )

    # Load pipeline once
    print(f"\nLoading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline(args)

    # Create processor
    processor = FontDiffuserBatchProcessor(args, pipe, font_manager)

    # Process Excel file
    print(f"\nProcessing Excel file: {args.excel_file}")
    results = processor.process_excel_file(
        excel_path=args.excel_file,
        base_output_dir=str(output_dir),
        style_image_path=args.style_image_path,
        generate_input_char=not args.skip_input_char,
        start_line=args.start_line,
        end_line=args.end_line,
    )

    return results


if __name__ == "__main__":
    try:
        results = main_batch_processing()
        print(f"\n✓ Batch processing completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during batch processing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

"""Example
python sample_excel.py \
    --excel_file "characters.xlsx" \
    --style_image_path "./style/A.png" \
    --ckpt_dir "./checkpoints" \
    --ttf_path "./fonts/KaiXinSongA.ttf" \
    --output_base_dir "./output_30_40" \
    --start_line 30 \
    --end_line 40 \
    --skip_input_char \
    --debug
"""

"""output/
├── batch_config.yaml
├── font_statistics.json
├── global_summary.json
├── debug/                          # Debug information
│   ├── content_images/            # All prepared content images
│   │   ├── content_A.png
│   │   ├── content_B.png
│   │   └── ...
│   └── rendering_issues.json      # Log of rendering issues
├── row_001_𠀖/
└── ...
"""
