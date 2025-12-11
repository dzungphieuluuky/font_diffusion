import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from pathlib import Path
import glob
import argparse # Added explicitly for arg_parse

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Import modules from the repository (Assuming these are available in your environment)
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

# =========================================================================
# Core Class with Style Caching Optimization
# =========================================================================

class FontDiffuserOptimized:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.cached_style_tensor = None  # Initialize cache
        
        print(f"🚀 Initializing FontDiffuser on {self.device} with FP16 Optimization...")
        
        # 1. Load Components
        unet = build_unet(args=args)
        unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location='cpu'))
        
        style_encoder = build_style_encoder(args=args)
        style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location='cpu'))
        
        content_encoder = build_content_encoder(args=args)
        content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location='cpu'))
        
        # 2. Build Model & Optimization (FP16)
        self.model = FontDiffuserModelDPM(
            unet=unet,
            style_encoder=style_encoder,
            content_encoder=content_encoder
        )
        
        self.model.half() 
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded and cast to FP16.")

        # 3. Load Scheduler
        train_scheduler = build_ddpm_scheduler(args=args)
        
        # 4. Build Pipeline
        self.pipe = FontDiffuserDPMPipeline(
            model=self.model,
            ddpm_train_scheduler=train_scheduler,
            model_type=args.model_type,
            guidance_type=args.guidance_type,
            guidance_scale=args.guidance_scale,
        )
        
        # 5. Define Transforms
        self.content_transform = transforms.Compose([
            transforms.Resize(args.content_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        self.style_transform = transforms.Compose([
            transforms.Resize(args.style_image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # 6. Cache Style Tensor (NEW OPTIMIZATION)
        if args.style_image_path:
            self._cache_style_tensor()
            
    def _cache_style_tensor(self):
        """Loads, transforms, and caches the style tensor to GPU memory."""
        args = self.args
        print(f"🖼️ Caching style image tensor from {args.style_image_path}...")
        try:
            style_image = Image.open(args.style_image_path).convert('RGB')
            style_tensor = self.style_transform(style_image)[None, :]
            # Cast to FP16 and move to device
            self.cached_style_tensor = style_tensor.to(self.device).half()
            print("✅ Style tensor cached successfully.")
        except Exception as e:
            print(f"❌ ERROR: Could not load or cache style tensor: {e}")
            self.cached_style_tensor = None


    def run_single_inference(self):
        """
        Handles single-image or single-character generation (original sample.py logic).
        Assumes content image path or character input is provided in args.
        """
        args = self.args
        
        if not args.style_image_path:
            raise ValueError("Style image path is required for inference.")
            
        if not args.content_image_path and not args.character_input:
             raise ValueError("Content image path or character input is required for single inference.")
             
        # Setup directories and seed
        os.makedirs(args.save_image_dir, exist_ok=True)
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

        if args.seed:
            set_seed(seed=args.seed)
        
        # --- Image Loading Logic ---
        content_image_pil = None
        
        # Load Content Image (File path OR Character generation)
        if args.character_input:
            assert args.content_character is not None, "The content_character should not be None."
            if not is_char_in_font(font_path=args.ttf_path, char=args.content_character):
                print(f"Character {args.content_character} not found in font.")
                return None
            font = load_ttf(ttf_path=args.ttf_path)
            content_image = ttf2im(font=font, char=args.content_character)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
        
        # Transform and Batch
        content_tensor = self.content_transform(content_image)[None, :]

        # --- Inference ---
        with torch.no_grad():
            content_tensor = content_tensor.to(self.device).half()
            
            print(f"🎨 Sampling single image by DPM-Solver++ (FP16 Mode)...")
            start = time.time()
            
            images = self.pipe.generate(
                content_images=content_tensor,
                style_images=self.cached_style_tensor, # Use cached style tensor
                batch_size=1,
                order=args.order,
                num_inference_step=args.num_inference_steps,
                content_encoder_downsample_size=args.content_encoder_downsample_size,
                t_start=args.t_start,
                t_end=args.t_end,
                dm_size=args.content_image_size,
                algorithm_type=args.algorithm_type,
                skip_type=args.skip_type,
                method=args.method,
                correcting_x0_fn=args.correcting_x0_fn
            )
            end = time.time()

            # --- Saving ---
            print(f"💾 Saving images to {args.save_image_dir}...")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            # Save comparison grid
            save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"✅ Finished. Time taken: {end - start:.4f}s")
            return images[0]


    def run_batch_inference(self):
        """
        Performs inference in batch mode by iterating over content images in a directory.
        """
        args = self.args
        
        if self.cached_style_tensor is None:
            raise RuntimeError("Style tensor must be cached before running batch inference.")

        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n--- Starting Batch Inference ---")
        print(f"Content Source: {args.content_dir}")
        print(f"Style Source: (CACHED TENSOR)")
        print(f"Output Target: {args.output_dir}")
        
        # Find all image files
        image_files = [f for f in os.listdir(args.content_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"No image files found in {args.content_dir}. Exiting.")
            return
            
        print(f"Found {len(image_files)} content images to process.")

        # Batch size for the diffusion pipeline (can be > 1 if memory allows)
        # For simplicity and robust memory usage, we keep it at 1 in this loop
        batch_size = 1 
        
        for i, image_file in enumerate(sorted(image_files)):
            full_content_path = os.path.join(args.content_dir, image_file)
            
            try:
                # 1. Load Content Image
                content_image = Image.open(full_content_path).convert('RGB')
                content_tensor = self.content_transform(content_image)[None, :]
                content_tensor = content_tensor.to(self.device).half()
                
                # 2. Inference
                start = time.time()
                with torch.no_grad():
                     images = self.pipe.generate(
                        content_images=content_tensor,
                        style_images=self.cached_style_tensor.repeat(batch_size, 1, 1, 1), # Repeat cached tensor
                        batch_size=batch_size, 
                        order=args.order,
                        num_inference_step=args.num_inference_steps,
                        content_encoder_downsample_size=args.content_encoder_downsample_size,
                        t_start=args.t_start,
                        t_end=args.t_end,
                        dm_size=args.content_image_size,
                        algorithm_type=args.algorithm_type,
                        skip_type=args.skip_type,
                        method=args.method,
                        correcting_x0_fn=args.correcting_x0_fn
                     )
                end = time.time()
                
                # 3. Saving
                output_name = Path(image_file).stem + "_stylized.png"
                output_path = os.path.join(args.output_dir, output_name)
                
                images[0].save(output_path) 
                
                print(f"[{i+1}/{len(image_files)}] Processed '{image_file}' (Time: {end - start:.2f}s) -> Saved to {output_path}")

            except Exception as e:
                print(f"❌ Error processing {image_file}: {e}")
                continue
                
# =========================================================================
# Argument Parsing and Main Execution
# =========================================================================

def arg_parse():
    # Assuming get_parser is available from your configs
    from configs.fontdiffuser import get_parser 

    parser = get_parser()
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None)
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image_dir", type=str, default="single_output",
                        help="The saving directory for single inference.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf")
    
    # === NEW BATCH ARGUMENTS ===
    parser.add_argument("--content_dir", type=str, default=None,
                        help="Directory containing content images for batch processing.")
    parser.add_argument("--output_dir", type=str, default="batch_output",
                        help="Output directory for batch processing results.")
    # ===========================
    
    args = parser.parse_args()
    
    # Fix tuple sizes for transforms
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)
    
    # Validation for single mode (set save_image = True if using single mode)
    args.save_image = bool(args.content_image_path or args.character_input) and not args.content_dir

    return args

if __name__=="__main__":
    args = arg_parse()
    
    # 1. Check if running in batch mode
    if args.content_dir and args.style_image_path:
        # Batch Mode: Initialize and run batch processor
        enhancer = FontDiffuserOptimized(args)
        enhancer.run_batch_inference()

    # 2. Check if running in single file/char mode
    elif (args.content_image_path or args.character_input) and args.style_image_path:
        # Single Mode: Initialize and run single inference
        enhancer = FontDiffuserOptimized(args)
        enhancer.run_single_inference()
        
    else:
        # Help message
        print("\n--- ERROR: Missing required arguments ---")
        print("To run in BATCH mode, you need:")
        print("  --style_image_path <path/to/style.png>")
        print("  --content_dir <path/to/content/images/>")
        print("  --output_dir <path/to/save/results/>")
        print("\nTo run in SINGLE mode, you need:")
        print("  --style_image_path <path/to/style.png>")
        print("  --content_image_path <path/to/content.png> OR --character_input --content_character 'A'")
        print("\nUse -h or --help for full argument list.")