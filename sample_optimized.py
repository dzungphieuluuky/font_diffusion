"""
Optimized sampling for FontDiffuser with SAFE optimizations
- Multi-character batch processing
- Multi-font support
- Performance optimizations without changing model architecture
"""

import os
import cv2
import time
import random
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from functools import lru_cache

import torch
import torchvision.transforms as transforms
from accelerate.utils import set_seed

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


def arg_parse():
    from configs.fontdiffuser import get_parser

    parser = get_parser()
    
    # Original arguments
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None,
                        help="Single character or comma-separated list")
    parser.add_argument("--content_image_path", type=str, default=None)
    parser.add_argument("--style_image_path", type=str, default=None)
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--save_image_dir", type=str, default=None,
                        help="The saving directory.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ttf_path", type=str, default="ttf/KaiXinSongA.ttf",
                        help="Path to single TTF file or directory with multiple fonts")
    
    # SAFE optimization arguments (don't change model architecture)
    parser.add_argument("--fp16", action="store_true", default=False,
                       help="Use FP16 precision (SAFE - applied after loading weights)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing multiple characters")
    parser.add_argument("--channels_last", action="store_true", default=False,
                       help="Use channels-last memory format (SAFE)")
    parser.add_argument("--deterministic", action="store_true", default=False,
                       help="Use deterministic algorithms for reproducibility")
    
    args = parser.parse_args()
    
    style_image_size = args.style_image_size
    content_image_size = args.content_image_size
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


class FontManager:
    """Manages single or multiple font files"""
    
    def __init__(self, ttf_path: str):
        self.fonts = {}
        self.font_paths = []
        self._load_fonts(ttf_path)
    
    def _load_fonts(self, ttf_path: str):
        """Load font(s) from path"""
        if os.path.isfile(ttf_path):
            # Single font file
            self.font_paths = [ttf_path]
            font_name = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                'path': ttf_path,
                'font': None,  # Lazy load
                'name': font_name
            }
            print(f"✓ Font loaded: {font_name}")
            
        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions = {'.ttf', '.otf', '.TTF', '.OTF'}
            font_files = [
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_extensions
            ]
            
            if not font_files:
                raise ValueError(f"No font files found in directory: {ttf_path}")
            
            self.font_paths = sorted(font_files)
            
            print(f"\n{'='*60}")
            print(f"Loading {len(font_files)} fonts from directory...")
            print('='*60)
            
            for font_path in self.font_paths:
                font_name = os.path.splitext(os.path.basename(font_path))[0]
                self.fonts[font_name] = {
                    'path': font_path,
                    'font': None,  # Lazy load
                    'name': font_name
                }
                print(f"✓ {font_name}")
            
            print('='*60)
            print(f"Loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")
    
    def get_font_names(self) -> List[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())
    
    @lru_cache(maxsize=32)
    def get_font(self, font_name: str):
        """Get font object by name (cached)"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        
        # Lazy load font
        if self.fonts[font_name]['font'] is None:
            self.fonts[font_name]['font'] = load_ttf(self.fonts[font_name]['path'])
        
        return self.fonts[font_name]['font']
    
    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]['path']
    
    @lru_cache(maxsize=1024)
    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font (cached)"""
        font_path = self.get_font_path(font_name)
        return is_char_in_font(font_path=font_path, char=char)
    
    def get_available_chars_for_font(self, font_name: str, 
                                     characters: List[str]) -> List[str]:
        """Get list of characters available in specific font"""
        return [
            char for char in characters 
            if self.is_char_in_font(font_name, char)
        ]


def parse_characters(content_character: str) -> List[str]:
    """
    Parse character input - supports single char or comma-separated list
    
    Args:
        content_character: Single character or comma-separated list
    
    Returns:
        List of individual characters
    """
    if not content_character:
        return []
    
    # Check if comma-separated
    if ',' in content_character:
        chars = [c.strip() for c in content_character.split(',') if c.strip()]
    else:
        # Single character or string of characters
        chars = [content_character.strip()] if len(content_character.strip()) == 1 else list(content_character.strip())
    
    return chars


def get_content_transform(content_image_size: tuple):
    """Cached content transform"""
    return transforms.Compose([
        transforms.Resize(content_image_size, 
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def get_style_transform(style_image_size: tuple):
    """Cached style transform"""
    return transforms.Compose([
        transforms.Resize(style_image_size,
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def image_process_batch(args, characters: List[str], font_manager: FontManager, 
                       font_name: str, style_image_path: str) -> Tuple:
    """
    Process multiple characters in batch
    
    Args:
        args: Arguments
        characters: List of characters to process
        font_manager: FontManager instance
        font_name: Name of font to use
        style_image_path: Path to style image
    
    Returns:
        Tuple of (content_batch, style_batch, content_pils, valid_chars)
    """
    # Load style image
    style_image = Image.open(style_image_path).convert('RGB')
    style_transform = get_style_transform(args.style_image_size)
    
    # Get font
    font = font_manager.get_font(font_name)
    content_transform = get_content_transform(args.content_image_size)
    
    # Get available characters
    available_chars = font_manager.get_available_chars_for_font(font_name, characters)
    
    if not available_chars:
        print(f"Warning: No characters available in font '{font_name}'")
        return None, None, None, None
    
    # Generate content images
    content_images = []
    content_images_pil = []
    
    for char in available_chars:
        try:
            content_image = ttf2im(font=font, char=char)
            content_images_pil.append(content_image.copy())
            content_images.append(content_transform(content_image))
        except Exception as e:
            print(f"Error processing character '{char}': {e}")
            continue
    
    if not content_images:
        return None, None, None, None
    
    # Stack into batch
    content_batch = torch.stack(content_images)
    style_batch = style_transform(style_image)[None, :].repeat(len(content_images), 1, 1, 1)
    
    return content_batch, style_batch, content_images_pil, available_chars


def image_process(args, content_image=None, style_image=None, 
                 content_character=None, font_manager=None, font_name=None):
    """
    Original single-image processing (maintained for compatibility)
    """
    char_to_use = content_character if content_character else args.content_character
    
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert char_to_use is not None, "The content_character should not be None."
            
            if font_manager and font_name:
                if not font_manager.is_char_in_font(font_name, char_to_use):
                    return None, None, None
                font = font_manager.get_font(font_name)
            else:
                if not is_char_in_font(font_path=args.ttf_path, char=char_to_use):
                    return None, None, None
                font = load_ttf(ttf_path=args.ttf_path)
            
            content_image = ttf2im(font=font, char=char_to_use)
            content_image_pil = content_image.copy()
        else:
            content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil = None
        
        style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert char_to_use is not None, "The content_character should not be None."
            
            if font_manager and font_name:
                if not font_manager.is_char_in_font(font_name, char_to_use):
                    return None, None, None
                font = font_manager.get_font(font_name)
            else:
                if not is_char_in_font(font_path=args.ttf_path, char=char_to_use):
                    return None, None, None
                font = load_ttf(ttf_path=args.ttf_path)
            
            content_image = ttf2im(font=font, char=char_to_use)
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil = None
        
    # Use cached transforms
    content_transform = get_content_transform(args.content_image_size)
    style_transform = get_style_transform(args.style_image_size)
    
    content_image = content_transform(content_image)[None, :]
    style_image = style_transform(style_image)[None, :]

    return content_image, style_image, content_image_pil


def load_fontdiffuser_pipeline(args):
    """
    Load FontDiffuser pipeline with SAFE optimizations
    Only applies optimizations that don't change model architecture
    """
    print("Loading FontDiffuser pipeline...")
    
    # Load the model state_dict (original architecture preserved)
    unet = build_unet(args=args)
    unet.load_state_dict(torch.load(f"{args.ckpt_dir}/unet.pth", map_location='cpu'))
    
    style_encoder = build_style_encoder(args=args)
    style_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/style_encoder.pth", map_location='cpu'))
    
    content_encoder = build_content_encoder(args=args)
    content_encoder.load_state_dict(torch.load(f"{args.ckpt_dir}/content_encoder.pth", map_location='cpu'))
    
    print("✓ Loaded model state_dict successfully")
    
    # SAFE: Apply FP16 conversion AFTER loading weights
    if args.fp16:
        print("Converting to FP16 precision...")
        unet = unet.half()
        style_encoder = style_encoder.half()
        content_encoder = content_encoder.half()
    
    # SAFE: Apply channels-last memory format (doesn't change computation)
    if args.channels_last:
        print("Converting to channels-last memory format...")
        unet = unet.to(memory_format=torch.channels_last)
        style_encoder = style_encoder.to(memory_format=torch.channels_last)
        content_encoder = content_encoder.to(memory_format=torch.channels_last)
    
    # Create model
    model = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    
    # Move to device with proper dtype
    dtype = torch.float16 if args.fp16 else torch.float32
    model.to(args.device, dtype=dtype)
    model.eval()  # Set to evaluation mode
    
    print("✓ Model moved to device")

    # Load the training ddpm_scheduler
    train_scheduler = build_ddpm_scheduler(args=args)
    print("✓ Loaded training DDPM scheduler successfully")

    # Load the DPM_Solver to generate the sample
    pipe = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=args.model_type,
        guidance_type=args.guidance_type,
        guidance_scale=args.guidance_scale,
    )
    print("✓ Loaded DPM-Solver pipeline successfully")

    return pipe


def sampling_batch(args, pipe, characters: List[str], font_manager: FontManager,
                  font_name: str, style_image_path: str) -> Tuple[List, List, float]:
    """
    Batch sampling for multiple characters with single font and style
    
    Returns:
        Tuple of (images, valid_chars, inference_time)
    """
    # Process images in batch
    content_batch, style_batch, content_pils, valid_chars = image_process_batch(
        args, characters, font_manager, font_name, style_image_path
    )
    
    if content_batch is None:
        return None, None, 0.0
    
    # Set seed for reproducibility
    if args.seed:
        set_seed(seed=args.seed)
    
    with torch.no_grad():
        dtype = torch.float16 if args.fp16 else torch.float32
        content_batch = content_batch.to(args.device, dtype=dtype)
        style_batch = style_batch.to(args.device, dtype=dtype)
        
        # Apply channels-last if enabled
        if args.channels_last:
            content_batch = content_batch.to(memory_format=torch.channels_last)
            style_batch = style_batch.to(memory_format=torch.channels_last)
        
        print(f"Sampling {len(valid_chars)} characters with DPM-Solver++ ...")
        start = time.time()
        
        # Process in batches
        all_images = []
        batch_size = args.batch_size
        
        for i in range(0, len(content_batch), batch_size):
            batch_content = content_batch[i:i+batch_size]
            batch_style = style_batch[i:i+batch_size]
            
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
                correcting_x0_fn=args.correcting_x0_fn
            )
            
            all_images.extend(images)
        
        end = time.time()
        inference_time = end - start
        
        print(f"✓ Generated {len(all_images)} images in {inference_time:.2f}s ({inference_time/len(all_images):.3f}s/img)")
        
        return all_images, valid_chars, inference_time


def sampling(args, pipe, content_image=None, style_image=None, 
            font_manager=None, font_name=None):
    """
    Original single-image sampling (maintained for compatibility)
    """
    if not args.demo:
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if args.seed:
        set_seed(seed=args.seed)
    
    content_image, style_image, content_image_pil = image_process(
        args=args, 
        content_image=content_image, 
        style_image=style_image,
        font_manager=font_manager,
        font_name=font_name
    )
    
    if content_image is None:
        print(f"The content_character you provided is not in the ttf. "
              f"Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        dtype = torch.float16 if args.fp16 else torch.float32
        content_image = content_image.to(args.device, dtype=dtype)
        style_image = style_image.to(args.device, dtype=dtype)
        
        # Apply channels-last if enabled
        if args.channels_last:
            content_image = content_image.to(memory_format=torch.channels_last)
            style_image = style_image.to(memory_format=torch.channels_last)
        
        print(f"Sampling by DPM-Solver++ ......")
        start = time.time()
        images = pipe.generate(
            content_images=content_image,
            style_images=style_image,
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
            correcting_x0_fn=args.correcting_x0_fn)
        end = time.time()

        if args.save_image:
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if args.character_input:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=args.resolution)
            print(f"Finish the sampling process, costing time {end - start}s")
        
        return images[0]


def main():
    args = arg_parse()
    
    print("\n" + "="*60)
    print("FONTDIFFUSER - OPTIMIZED SAMPLING")
    print("="*60)
    print(f"Model: {args.ckpt_dir}")
    print(f"Device: {args.device}")
    print(f"FP16: {args.fp16}")
    print(f"Channels Last: {args.channels_last}")
    print(f"Batch Size: {args.batch_size}")
    print("="*60 + "\n")
    
    # Load pipeline
    pipe = load_fontdiffuser_pipeline(args=args)
    
    # Check if multi-character or multi-font mode
    if args.character_input and args.content_character:
        characters = parse_characters(args.content_character)
        
        if len(characters) > 1 or os.path.isdir(args.ttf_path):
            # Multi-character or multi-font mode
            print(f"\n{'='*60}")
            print("BATCH MODE ACTIVATED")
            print(f"Characters: {len(characters)}")
            print('='*60)
            
            # Load font manager
            font_manager = FontManager(args.ttf_path)
            font_names = font_manager.get_font_names()
            
            if not args.demo:
                os.makedirs(args.save_image_dir, exist_ok=True)
                save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")
            
            # Process each font
            for font_idx, font_name in enumerate(font_names):
                print(f"\n[Font {font_idx+1}/{len(font_names)}] {font_name}")
                print("-" * 60)
                
                # Get available characters
                available = font_manager.get_available_chars_for_font(font_name, characters)
                print(f"Available characters: {len(available)}/{len(characters)}")
                
                if not available:
                    print("Skipping font (no characters available)")
                    continue
                
                # Sample in batch
                images, valid_chars, inf_time = sampling_batch(
                    args, pipe, characters, font_manager, 
                    font_name, args.style_image_path
                )
                
                if images is None:
                    continue
                
                # Save images
                if args.save_image:
                    font_dir = os.path.join(args.save_image_dir, font_name)
                    os.makedirs(font_dir, exist_ok=True)
                    
                    for char, img in zip(valid_chars, images):
                        char_dir = os.path.join(font_dir, char)
                        os.makedirs(char_dir, exist_ok=True)
                        save_single_image(save_dir=char_dir, image=img)
                    
                    print(f"✓ Saved {len(images)} images to {font_dir}")
            
            print("\n" + "="*60)
            print("✓ BATCH PROCESSING COMPLETE")
            print("="*60)
        else:
            # Single character mode
            out_image = sampling(args=args, pipe=pipe)
    else:
        # Original single-image mode
        out_image = sampling(args=args, pipe=pipe)


if __name__=="__main__":
    main()


"""
USAGE EXAMPLES:

1. Single Character (Original Mode):
python sample_optimized_safe.py \
    --ckpt_dir="ckpt/" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --character_input \
    --content_character="隆" \
    --save_image_dir="outputs/" \
    --device="cuda:0" \
    --algorithm_type="dpmsolver++" \
    --guidance_type="classifier-free" \
    --guidance_scale=7.5 \
    --num_inference_steps=20 \
    --method="multistep" \
    --fp16 \
    --channels_last

2. Multiple Characters (Batch Mode):
python sample_optimized_safe.py \
    --ckpt_dir="ckpt/" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --character_input \
    --content_character="漢,字,書,法,藝,術" \
    --save_image_dir="outputs/" \
    --device="cuda:0" \
    --batch_size=4 \
    --fp16 \
    --channels_last

3. Multiple Fonts (Directory Mode):
python sample_optimized_safe.py \
    --ckpt_dir="ckpt/" \
    --ttf_path="ttf/" \
    --style_image_path="data_examples/sampling/example_style.jpg" \
    --save_image \
    --character_input \
    --content_character="漢,字,書" \
    --save_image_dir="outputs/" \
    --device="cuda:0" \
    --batch_size=4 \
    --fp16 \
    --channels_last

4. Single Font + Multiple Characters + Max Speed:
python sample_optimized_safe.py \
    --ckpt_dir="ckpt/" \
    --ttf_path="ttf/MyFont.ttf" \
    --style_image_path="style.jpg" \
    --save_image \
    --character_input \
    --content_character="漢,字,書,法,藝,術,文,化,學,習" \
    --save_image_dir="outputs/" \
    --batch_size=8 \
    --num_inference_steps=15 \
    --fp16 \
    --channels_last
"""