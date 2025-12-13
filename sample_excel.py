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
warnings.filterwarnings('ignore')

# Import FontDiffuser components
try:
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM,
        build_ddpm_scheduler,
        build_unet,
        build_content_encoder,
        build_style_encoder
    )
    from utils import (
        save_args_to_yaml,
        save_single_image,
        save_image_with_content_style
    )

    from font_manager import (
        FontManager,
        FontRenderer
    )
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
            'characters_processed': 0,
            'characters_skipped_no_font': 0,
            'characters_skipped_render_failed': 0,
            'generation_errors': 0,
            'fonts_used': defaultdict(int),
            'edge_cases_fixed': 0
        }
    
    def _create_transforms(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """Create image transforms for content and style images"""
        content_transforms = transforms.Compose([
            transforms.Resize(
                self.args.content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        style_transforms = transforms.Compose([
            transforms.Resize(
                self.args.style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return content_transforms, style_transforms
    
    def prepare_images(self, char: str, style_image_path: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Image.Image]]:
        """
        Prepare content and style images for a character with robust error handling
        """
        try:
            # Load style image
            if not os.path.exists(style_image_path):
                raise FileNotFoundError(f"Style image not found: {style_image_path}")
            
            style_image = Image.open(style_image_path).convert('RGB')
            style_image_tensor = self.style_transforms(style_image)[None, :].to(self.device)
            
            # Prepare content image using font manager
            if self.args.character_input:
                # Check font support
                if not self.font_manager.can_render_character(char):
                    print(f"    Character '{char}' (U+{ord(char):04X}) not supported by any font")
                    self.stats['characters_skipped_no_font'] += 1
                    return None, None, None
                
                # Render character
                content_image = self.font_manager.render_character(char)
                if content_image is None:
                    print(f"    Failed to render character '{char}'")
                    self.stats['characters_skipped_render_failed'] += 1
                    return None, None, None
                
                # Check rendering quality
                img_array = np.array(content_image.convert('L'))
                
                # Detect potential rendering issues
                issues = self._detect_rendering_issues(img_array, char)
                
                if issues:
                    print(f"    Rendering issues for '{char}': {', '.join(issues)}")
                    
                    # Try to fix common issues
                    if 'edge_touching' in issues:
                        content_image = self._fix_edge_touching(content_image)
                        self.stats['edge_cases_fixed'] += 1
                
                content_image_pil = content_image.copy()
                
                # Track which font was used
                font_info = self.font_manager.get_supporting_font(char)
                if font_info:
                    self.stats['fonts_used'][font_info['name']] += 1
            else:
                # If not using character input, use white image
                content_image = Image.new('RGB', (256, 256), color='white')
                content_image_pil = None
            
            # Convert content image to tensor
            content_image_tensor = self.content_transforms(content_image)[None, :].to(self.device)
            
            # Debug: Save the prepared content image
            if hasattr(self.args, 'debug') and self.args.debug:
                debug_dir = Path(self.args.output_base_dir) / "debug" / "content_images"
                debug_dir.mkdir(parents=True, exist_ok=True)
                content_image.save(debug_dir / f"content_{char}.png")
            
            return content_image_tensor, style_image_tensor, content_image_pil
            
        except Exception as e:
            print(f"    Error preparing images for '{char}': {e}")
            self.stats['generation_errors'] += 1
            return None, None, None
    
    def _detect_rendering_issues(self, img_array: np.ndarray, char: str) -> List[str]:
        """Detect potential rendering issues in character image"""
        issues = []
        
        # Check if image is mostly one color
        if np.std(img_array) < 10:
            issues.append('low_contrast')
        
        # Check if character touches edges
        edge_thickness = 3
        edges = [
            img_array[:edge_thickness, :],  # Top
            img_array[-edge_thickness:, :],  # Bottom
            img_array[:, :edge_thickness],   # Left
            img_array[:, -edge_thickness:]   # Right
        ]
        
        edge_threshold = 200
        for i, edge in enumerate(edges):
            if np.any(edge < edge_threshold):
                issues.append('edge_touching')
                break
        
        # Check if character is too small
        binary = img_array < 240  # Threshold for character pixels
        character_pixels = np.sum(binary)
        total_pixels = img_array.size
        
        if character_pixels < total_pixels * 0.01:  # Less than 1% of pixels
            issues.append('too_small')
        
        return issues
    
    def _fix_edge_touching(self, image: Image.Image) -> Image.Image:
        """Fix edge-touching by adding padding"""
        # Convert to numpy
        img_array = np.array(image.convert('RGB'))
        
        # Add white border
        border_size = 20
        h, w = img_array.shape[:2]
        new_h, new_w = h + border_size * 2, w + border_size * 2
        
        new_array = np.ones((new_h, new_w, 3), dtype=np.uint8) * 255
        new_array[border_size:border_size+h, border_size:border_size+w] = img_array
        
        return Image.fromarray(new_array).resize((h, w), Image.LANCZOS)
    
    def generate_character(self, char: str, style_image_path: str) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
        """Generate a single character using FontDiffuser"""
        # Set seed for reproducibility if specified
        if self.args.seed:
            set_seed(seed=self.args.seed)
        
        # Prepare images
        content_tensor, style_tensor, content_pil = self.prepare_images(char, style_image_path)
        
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
                    correcting_x0_fn=self.args.correcting_x0_fn
                )
            
            self.stats['characters_processed'] += 1
            return images[0], content_pil
            
        except Exception as e:
            print(f"    Error generating character '{char}': {e}")
            self.stats['generation_errors'] += 1
            return None, None

def parse_excel_batch_args():
    """Parse arguments for batch processing with multi-font support"""
    from configs.fontdiffuser import get_parser
    
    parser = get_parser()
    
    # Existing arguments
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--character_input", action="store_true", default=True)
    parser.add_argument("--style_image_path", type=str, required=True, 
                       help="Path to style reference image")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    # Font arguments
    parser.add_argument("--ttf_path", type=str, action='append', 
                       help="Path to TTF font file (can be specified multiple times)")
    parser.add_argument("--font_dir", type=str, default=None,
                       help="Directory containing multiple TTF fonts (all .ttf files will be loaded)")
    
    # Batch processing arguments
    parser.add_argument("--excel_file", type=str, required=True,
                       help="Path to Excel file with character data")
    parser.add_argument("--output_base_dir", type=str, default="./fontdiffuser_batch_output",
                       help="Base directory for all outputs")
    parser.add_argument("--skip_input_char", action="store_true",
                       help="Skip generating the input character")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Maximum number of rows to process")
    parser.add_argument("--debug", action="store_true",
                   help="Save debug images and additional information")

    
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
        font_paths.extend([
            str(p) for p in Path(args.font_dir).glob("*.ttf")
        ])
        font_paths.extend([
            str(p) for p in Path(args.font_dir).glob("*.TTF")
        ])
    
    # Remove duplicates
    font_paths = list(dict.fromkeys(font_paths))
    
    if not font_paths:
        # Try default location
        default_font = Path(__file__).parent / "fonts" / "default.ttf"
        if default_font.exists():
            font_paths = [str(default_font)]
        else:
            raise ValueError("No font files specified or found. Use --ttf_path or --font_dir")
    
    args.font_paths = font_paths
    
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
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
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
    """Main function for batch processing Excel file with multi-font support"""
    args = parse_excel_batch_args()
    
    print(f"\n{'='*70}")
    print(f"FONTDIFFUSER BATCH PROCESSING")
    print(f"{'='*70}")
    
    # Create output directory
    output_dir = Path(args.output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    save_args_to_yaml(
        args=args, 
        output_file=str(output_dir / "batch_config.yaml")
    )
    
    # Initialize font manager with multiple fonts
    print(f"\nLoading fonts...")
    font_manager = FontManager(
        font_paths=args.font_paths,
        font_size=128  # You can make this configurable if needed
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
        max_rows=args.max_rows
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
    --output_base_dir "./output" \
    --debug  # Enable debug mode
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