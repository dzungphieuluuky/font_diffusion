import os
import pandas as pd
import ast
from typing import List, Dict, Tuple
import time
from pathlib import Path

import os
import cv2
import time
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Add to your existing imports
import warnings
warnings.filterwarnings('ignore')

from src import (FontDiffuserDPMPipeline,
                 FontDiffuserModelDPM,
                 build_ddpm_scheduler,
                 build_unet,
                 build_content_encoder,
                 build_style_encoder)
from utils import (ttf2im,
                   load_ttf,
                   is_char_in_font,
                   save_args_to_yaml,
                   save_single_image,
                   save_image_with_content_style)

class FontDiffuserBatchProcessor:
    """Process Excel file and generate fonts for similar characters"""
    
    def __init__(self, args, pipe):
        self.args = args
        self.pipe = pipe
        self.device = args.device
        
        # Create transforms
        self.content_transforms, self.style_transforms = self._create_transforms()
        
        # Initialize font if needed
        self.ttf_font = None
        if args.character_input and args.ttf_path:
            from utils import load_ttf
            self.ttf_font = load_ttf(ttf_path=args.ttf_path)
    
    def _create_transforms(self):
        """Create image transforms"""
        content_transforms = transforms.Compose([
            transforms.Resize(self.args.content_image_size, 
                             interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        style_transforms = transforms.Compose([
            transforms.Resize(self.args.style_image_size, 
                             interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        return content_transforms, style_transforms
    
    def prepare_images(self, input_char: str, style_image_path: str):
        """Prepare content and style images for a character"""
        from utils import is_char_in_font, ttf2im
        from PIL import Image
        
        # Load style image
        if not os.path.exists(style_image_path):
            raise FileNotFoundError(f"Style image not found: {style_image_path}")
        
        style_image = Image.open(style_image_path).convert('RGB')
        style_image_tensor = self.style_transforms(style_image)[None, :].to(self.device)
        
        # Prepare content image
        if self.args.character_input:
            if not is_char_in_font(font_path=self.args.ttf_path, char=input_char):
                print(f"Character '{input_char}' not in font, skipping...")
                return None, None, None
            
            content_image = ttf2im(font=self.ttf_font, char=input_char)
            content_image_pil = content_image.copy()
        else:
            # If not using character input
            content_image = Image.new('RGB', (256, 256), color='white')
            content_image_pil = None
        
        content_image_tensor = self.content_transforms(content_image)[None, :].to(self.device)
        
        return content_image_tensor, style_image_tensor, content_image_pil
    
    def generate_character(self, char: str, style_image_path: str):
        """Generate a single character"""
        # Set seed for reproducibility
        if self.args.seed:
            from accelerate.utils import set_seed
            set_seed(seed=self.args.seed)
        
        # Prepare images
        content_tensor, style_tensor, content_pil = self.prepare_images(
            char, style_image_path
        )
        
        if content_tensor is None:
            return None
        
        # Generate
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
        
        return images[0], content_pil
    
    def process_excel_file(self, excel_path: str, 
                          base_output_dir: str,
                          style_image_path: str = None,
                          generate_input_char: bool = True):
        """
        Process Excel file and generate fonts for similar characters
        
        Args:
            excel_path: Path to Excel file
            base_output_dir: Base directory for outputs
            style_image_path: Path to style image (overrides args.style_image_path)
            generate_input_char: Whether to generate the input character itself
        """
        # Use provided style image or default from args
        if style_image_path is None:
            style_image_path = self.args.style_image_path
        
        # Load Excel file
        print(f"Loading Excel file: {excel_path}")
        df = pd.read_excel(excel_path)
        
        # Check required columns
        required_columns = ['Input Character', 'Top 20 Similar Characters']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        results_summary = {}
        
        # Process each row
        for idx, row in df.iterrows():
            input_char = str(row['Input Character']).strip()
            similar_chars_str = row['Top 20 Similar Characters']
            
            print(f"\n{'='*60}")
            print(f"Processing Row {idx+1}: Input Character = '{input_char}'")
            print(f"{'='*60}")
            
            # Parse similar characters (handle different formats)
            try:
                # Try to parse as Python list string
                if isinstance(similar_chars_str, str) and similar_chars_str.startswith('['):
                    similar_chars = ast.literal_eval(similar_chars_str)
                else:
                    # Try other formats
                    similar_chars = str(similar_chars_str).strip("[]").replace("'", "").split(',')
                    similar_chars = [c.strip() for c in similar_chars if c.strip()]
                
                # Limit to top 20
                similar_chars = similar_chars[:20]
                
            except Exception as e:
                print(f"Error parsing similar characters: {e}")
                similar_chars = []
            
            print(f"Found {len(similar_chars)} similar characters")
            
            # Create folder for this input character
            # Sanitize folder name (remove invalid characters)
            safe_char_name = self._sanitize_filename(input_char)
            char_output_dir = os.path.join(base_output_dir, f"char_{safe_char_name}")
            os.makedirs(char_output_dir, exist_ok=True)
            
            # Generate characters
            generated_chars = {}
            
            # Generate input character if requested
            if generate_input_char:
                print(f"\nGenerating input character: '{input_char}'")
                try:
                    result = self.generate_character(input_char, style_image_path)
                    if result:
                        generated_image, content_pil = result
                        generated_chars[input_char] = generated_image
                        
                        # Save input character
                        input_char_dir = os.path.join(char_output_dir, "input_character")
                        os.makedirs(input_char_dir, exist_ok=True)
                        
                        from utils import save_single_image, save_image_with_content_style
                        
                        save_single_image(
                            save_dir=input_char_dir,
                            image=generated_image
                        )
                        
                        if self.args.character_input:
                            save_image_with_content_style(
                                save_dir=input_char_dir,
                                image=generated_image,
                                content_image_pil=content_pil,
                                content_image_path=None,
                                style_image_path=style_image_path,
                                resolution=self.args.resolution
                            )
                        
                        print(f"✓ Generated input character '{input_char}'")
                    else:
                        print(f"✗ Failed to generate input character '{input_char}'")
                except Exception as e:
                    print(f"Error generating input character '{input_char}': {e}")
            
            # Generate similar characters
            print(f"\nGenerating similar characters:")
            for i, similar_char in enumerate(similar_chars, 1):
                print(f"  [{i}/{len(similar_chars)}] Character: '{similar_char}'", end="", flush=True)
                
                try:
                    result = self.generate_character(similar_char, style_image_path)
                    if result:
                        generated_image, content_pil = result
                        generated_chars[similar_char] = generated_image
                        
                        # Save similar character
                        similar_char_dir = os.path.join(char_output_dir, "similar_characters")
                        os.makedirs(similar_char_dir, exist_ok=True)
                        
                        # Save with character name in filename
                        safe_similar_name = self._sanitize_filename(similar_char)
                        char_specific_dir = os.path.join(similar_char_dir, safe_similar_name)
                        os.makedirs(char_specific_dir, exist_ok=True)
                        
                        from utils import save_single_image, save_image_with_content_style
                        
                        save_single_image(
                            save_dir=char_specific_dir,
                            image=generated_image
                        )
                        
                        if self.args.character_input:
                            save_image_with_content_style(
                                save_dir=char_specific_dir,
                                image=generated_image,
                                content_image_pil=content_pil,
                                content_image_path=None,
                                style_image_path=style_image_path,
                                resolution=self.args.resolution
                            )
                        
                        print(f" ✓")
                    else:
                        print(f" ✗ (Character not in font)")
                except Exception as e:
                    print(f" ✗ (Error: {str(e)[:50]})")
            
            # Save summary for this character
            results_summary[input_char] = {
                'output_dir': char_output_dir,
                'generated_count': len(generated_chars),
                'similar_characters': similar_chars,
                'generated_similar': list(generated_chars.keys())
            }
            
            # Create a summary file
            summary_path = os.path.join(char_output_dir, "generation_summary.txt")
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Input Character: {input_char}\n")
                f.write(f"Output Directory: {char_output_dir}\n")
                f.write(f"Total Generated: {len(generated_chars)}\n")
                f.write(f"Style Image: {style_image_path}\n")
                f.write(f"Generation Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Generated Characters:\n")
                for char_name in generated_chars.keys():
                    f.write(f"  - {char_name}\n")
                
                f.write("\nAll Similar Characters (from Excel):\n")
                for char_name in similar_chars:
                    status = "✓" if char_name in generated_chars else "✗"
                    f.write(f"  {status} {char_name}\n")
        
        # Save global summary
        global_summary_path = os.path.join(base_output_dir, "global_summary.txt")
        with open(global_summary_path, 'w', encoding='utf-8') as f:
            f.write(f"FontDiffuser Batch Processing Summary\n")
            f.write(f"====================================\n")
            f.write(f"Excel File: {excel_path}\n")
            f.write(f"Base Output Directory: {base_output_dir}\n")
            f.write(f"Total Input Characters Processed: {len(results_summary)}\n")
            f.write(f"Processing Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            total_generated = sum(info['generated_count'] for info in results_summary.values())
            f.write(f"Total Characters Generated: {total_generated}\n\n")
            
            f.write("Details by Input Character:\n")
            for input_char, info in results_summary.items():
                f.write(f"\n{'─'*40}\n")
                f.write(f"Input Character: {input_char}\n")
                f.write(f"Output Directory: {info['output_dir']}\n")
                f.write(f"Generated: {info['generated_count']} characters\n")
                f.write(f"Similar Characters: {len(info['similar_characters'])}\n")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total input characters processed: {len(results_summary)}")
        print(f"Output base directory: {base_output_dir}")
        print(f"Global summary saved to: {global_summary_path}")
        
        return results_summary
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize string to be safe for filenames"""
        # Replace problematic characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Also handle Unicode characters by encoding them
        import urllib.parse
        safe_name = urllib.parse.quote(filename, safe='')
        
        # Limit length
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name


def parse_excel_batch_args():
    """Parse arguments for batch processing"""
    from configs.fontdiffuser import get_parser
    
    parser = get_parser()
    
    # Existing arguments
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Checkpoint directory")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--character_input", action="store_true", default=True)
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, required=True, 
                       help="Path to style reference image")
    parser.add_argument("--save_image", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, required=True, help="Path to TTF font file")
    
    # New batch processing arguments
    parser.add_argument("--excel_file", type=str, required=True,
                       help="Path to Excel file with character data")
    parser.add_argument("--output_base_dir", type=str, default="./fontdiffuser_batch_output",
                       help="Base directory for all outputs")
    parser.add_argument("--skip_input_char", action="store_true",
                       help="Skip generating the input character (only generate similar ones)")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Maximum number of rows to process (for testing)")
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel processing (experimental)")
    
    args = parser.parse_args()
    
    # Set image sizes
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    
    # Set defaults
    if not hasattr(args, 'save_image_dir'):
        args.save_image_dir = args.output_base_dir
    
    return args

def load_fontdiffuser_pipeline(args):
    # Load the model state_dict
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth"))
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth"))
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth"))
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder)
    model.to(args.device)
    print("Loaded the model state_dict successfully!")

    # Load the training ddpm_scheduler.
    train_scheduler = build_ddpm_scheduler(args=args)
    print("Loaded training DDPM scheduler sucessfully!")

    # Load the DPM_Solver to generate the sample.
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("Loaded dpm_solver pipeline sucessfully!")

    return pipe

def main_batch_processing():
    """Main function for batch processing Excel file"""
    args = parse_excel_batch_args()
    
    # Create output base directory
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Save configuration
    from utils import save_args_to_yaml
    save_args_to_yaml(
        args=args, 
        output_file=f"{args.output_base_dir}/batch_config.yaml"
    )
    
    # Load pipeline once
    print("Loading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline(args=args)
    
    # Create processor
    processor = FontDiffuserBatchProcessor(args, pipe)
    
    # Process Excel file
    results = processor.process_excel_file(
        excel_path=args.excel_file,
        base_output_dir=args.output_base_dir,
        style_image_path=args.style_image_path,
        generate_input_char=not args.skip_input_char
    )
    
    return results


if __name__ == "__main__":
    # You can choose which mode to run
    
    # Option 1: Original single character mode
    # args = arg_parse()
    # pipe = load_fontdiffuser_pipeline(args=args)
    # out_image = sampling(args=args, pipe=pipe)
    
    # Option 2: Batch processing mode (recommended for your use case)
    results = main_batch_processing()

"""Example
python sample_excel.py \
    --excel_file "your_characters.xlsx" \
    --style_image_path "./style/A.png" \
    --ckpt_dir "./checkpoints" \
    --ttf_path "./fonts/default.ttf" \
    --output_base_dir "./font_generations" \
    --device "cuda:0"
"""

"""Expected Output File Structure
font_generations/
├── batch_config.yaml
├── global_summary.txt
│
├── char_𠀖/                    # Folder for first input character
│   ├── generation_summary.txt
│   ├── input_character/
│   │   └── 𠀖.png            # Generated input character
│   └── similar_characters/
│       ├── 共/                # Each similar character gets its own folder
│       │   └── 共.png
│       ├── 𠀗/
│       │   └── 𠀗.png
│       └── ... (20 folders)
│
├── char_𠀗/                    # Folder for second input character
│   ├── generation_summary.txt
│   ├── input_character/
│   │   └── 𠀗.png
│   └── similar_characters/
│       ├── 𠀖/
│       │   └── 𠀖.png
│       ├── 共/
│       │   └── 共.png
│       └── ...
│
└── ... (one folder per Excel row)
"""