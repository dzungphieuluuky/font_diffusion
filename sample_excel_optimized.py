"""
FontDiffuser Standard Layout Generator with Structured Logging
Generates training data in the official FontDiffuser format:
data_examples/train/ContentImage/  +  TargetImage/styleX/
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Import project-specific modules
from sample_optimized import load_fontdiffuser_pipeline_safe
from utils import save_args_to_yaml
from font_manager import FontManager


# ----------------------------------------------------------------------
# Logging Configuration
# ----------------------------------------------------------------------
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure structured logging for the application.

    :param str log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :param str | None log_file: Optional path to log file.
    :return logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger("FontDiffuserStandard")
    logger.setLevel(log_level)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ----------------------------------------------------------------------
# Main Processor Class
# ----------------------------------------------------------------------
class FontDiffuserStandardLayoutGenerator:
    """
    Generates FontDiffuser training data in the standard directory layout.
    """

    def __init__(self, args: argparse.Namespace, pipe: torch.nn.Module, font_manager: FontManager):
        """
        Initialize the generator.

        :param argparse.Namespace args: Parsed command-line arguments.
        :param torch.nn.Module pipe: Loaded FontDiffuser pipeline.
        :param FontManager font_manager: Font manager instance.
        """
        self.args = args
        self.pipe = pipe
        self.font_manager = font_manager
        self.logger = logging.getLogger("FontDiffuserStandard")

        # Transforms (cached via sample_optimized)
        self.content_transform = transforms.Compose([
            transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.style_transform = transforms.Compose([
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # Output directories
        self.base_dir = Path(args.output_base_dir) / "data_examples" / "train"
        self.content_dir = self.base_dir / "ContentImage"
        self.target_dir = self.base_dir / "TargetImage"

        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Mapping and counters
        self.char_to_index: Dict[str, int] = {}
        self.next_char_index = 0
        self.style_to_index: Dict[str, int] = {}
        self.next_style_index = 0

        self.stats = {
            "content_images": 0,
            "target_images": 0,
            "characters_skipped": 0,
            "generation_errors": 0,
        }

    def _get_char_index(self, char: str) -> int:
        """Assign sequential index to unique characters."""
        if char not in self.char_to_index:
            self.char_to_index[char] = self.next_char_index
            self.next_char_index += 1
        return self.char_to_index[char]

    def _get_style_index(self, style_path: str) -> int:
        """Assign sequential index to unique style images."""
        if style_path not in self.style_to_index:
            idx = self.next_style_index
            self.style_to_index[style_path] = idx
            (self.target_dir / f"style{idx}").mkdir(exist_ok=True)
            self.next_style_index += 1
        return self.style_to_index[style_path]

    def save_content_image(self, char: str) -> Optional[Path]:
        """Render and save deduplicated content image."""
        idx = self._get_char_index(char)
        path = self.content_dir / f"char{idx}.png"

        if path.exists():
            return path

        img = self.font_manager.render_character(char)
        if img is None:
            self.stats["characters_skipped"] += 1
            self.logger.warning("Character '%s' not renderable by any font", char)
            return None

        img.save(path)
        self.stats["content_images"] += 1
        return path

    def generate_target_image(self, char: str, style_path: str) -> bool:
        """Generate and save stylized target image."""
        char_idx = self._get_char_index(char)
        style_idx = self._get_style_index(style_path)
        style_dir = self.target_dir / f"style{style_idx}"
        target_path = style_dir / f"style{style_idx}+char{char_idx}.png"

        if target_path.exists():
            return True

        try:
            content_img = self.font_manager.render_character(char)
            if content_img is None:
                return False

            style_img = Image.open(style_path).convert("RGB")

            content_tensor = self.content_transform(content_img)[None, :]
            style_tensor = self.style_transform(style_img)[None, :]

            dtype = torch.float16 if self.args.fp16 else torch.float32
            content_tensor = content_tensor.to(self.args.device, dtype=dtype)
            style_tensor = style_tensor.to(self.args.device, dtype=dtype)

            if self.args.seed is not None:
                set_seed(self.args.seed)

            with torch.no_grad():
                images = self.pipe.generate(
                    content_images=content_tensor,
                    style_images=style_tensor,
                    batch_size=1,
                    num_inference_step=self.args.num_inference_steps,
                    order=self.args.order,
                    content_encoder_downsample_size=self.args.content_encoder_downsample_size,
                    t_start=self.args.t_start,
                    t_end=self.args.t_end,
                    dm_size=self.args.content_image_size,
                    algorithm_type=self.args.algorithm_type,
                    skip_type=self.args.skip_type,
                    method=self.args.method,
                    correcting_x0_fn=self.args.correcting_x0_fn,
                )

            images[0].save(target_path)
            self.stats["target_images"] += 1
            return True

        except Exception as e:
            self.logger.error("Generation failed for char '%s' with style '%s': %s", char, style_path, e)
            self.stats["generation_errors"] += 1
            return False

    def process_excel(self, excel_path: str, style_paths: List[str]) -> None:
        """Main processing loop."""
        self.logger.info("Starting standard layout generation")
        self.logger.info("Excel file: %s", excel_path)
        self.logger.info("Style images: %s", style_paths)

        df = pd.read_excel(excel_path)
        self.logger.info("Loaded %d rows from Excel", len(df))

        for _, row in df.iterrows():
            input_char = str(row["Input Character"]).strip()
            similar_str = row.get("Top 20 Similar Characters", "")
            similar_chars = self._parse_similar_chars(similar_str)

            chars = [input_char] + similar_chars if not self.args.skip_input_char else similar_chars
            self.logger.info("Processing input char '%s' with %d similar chars", input_char, len(similar_chars))

            for char in chars:
                self.logger.debug("Processing character: %s", char)

                if self.save_content_image(char) is None:
                    continue

                for style_path in style_paths:
                    success = self.generate_target_image(char, style_path)
                    self.logger.debug("Target for style '%s': %s", style_path, "success" if success else "failed")

        self._save_summary()

    def _parse_similar_chars(self, cell) -> List[str]:
        """Parse similar characters from Excel cell."""
        if pd.isna(cell):
            return []
        try:
            if isinstance(cell, str) and cell.startswith("["):
                import ast
                return ast.literal_eval(cell)[:20]
            return [c.strip() for c in str(cell).strip("[]'\" ").split(",") if c.strip()][:20]
        except Exception:
            return []

    def _save_summary(self) -> None:
        """Save generation summary and statistics."""
        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "output_directory": str(self.base_dir),
            "character_mapping": self.char_to_index,
            "style_mapping": self.style_to_index,
            "statistics": self.stats,
        }
        summary_path = self.base_dir / "generation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        self.logger.info("Generation summary saved to %s", summary_path)


# ----------------------------------------------------------------------
# Argument Parsing & Main
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments with key parameters."""
    from configs.fontdiffuser import get_parser

    parser = get_parser()

    parser.add_argument("--excel_file", type=str, required=True, help="Path to Excel file")
    parser.add_argument("--style_image_path", type=str, action="append", required=True,
                        help="One or more style reference images (can be repeated)")
    parser.add_argument("--output_base_dir", type=str, default="fontdiffuser_dataset",
                        help="Base output directory")
    parser.add_argument("--font_dir", type=str, required=True, help="Directory with TTF fonts")
    parser.add_argument("--skip_input_char", action="store_true", help="Skip generating input character")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path")

    args = parser.parse_args()

    # Ensure tuple sizes
    s = args.style_image_size
    c = args.content_image_size
    args.style_image_size = (s, s)
    args.content_image_size = (c, c)

    return args


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup logging
    logger = setup_logging(log_level=args.log_level, log_file=args.log_file)
    logger.info("=" * 70)
    logger.info("FONTDIFFUSER STANDARD LAYOUT GENERATOR")
    logger.info("=" * 70)

    # Log important parameters
    logger.info("Key Parameters:")
    logger.info("  Checkpoint: %s", args.ckpt_dir)
    logger.info("  Device: %s", args.device)
    logger.info("  FP16: %s", args.fp16)
    logger.info("  Inference steps: %d", args.num_inference_steps)
    logger.info("  Guidance scale: %.1f", args.guidance_scale)
    logger.info("  Batch size: %d", getattr(args, "batch_size", 1))
    logger.info("  torch.compile: %s", getattr(args, "compile", False))
    logger.info("  Channels last: %s", getattr(args, "channels_last", True))
    logger.info("  Seed: %s", args.seed if args.seed else "None")
    logger.info("  Output base: %s", args.output_base_dir)
    logger.info("  Styles: %d image(s)", len(args.style_image_path))
    logger.info("=" * 70)

    # Save config
    save_args_to_yaml(args, str(Path(args.output_base_dir) / "run_config.yaml"))

    # Initialize components
    font_manager = FontManager(font_dir=args.font_dir)
    pipe = load_fontdiffuser_pipeline_safe(args)

    # Run generation
    generator = FontDiffuserStandardLayoutGenerator(args, pipe, font_manager)
    generator.process_excel(
        excel_path=args.excel_file,
        style_paths=args.style_image_path,
    )

    logger.info("Generation completed successfully!")
    logger.info("Output location: %s", generator.base_dir)


if __name__ == "__main__":
    main()