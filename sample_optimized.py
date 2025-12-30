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
from typing import List, Optional, Tuple, Dict, Any, Union
from functools import lru_cache
from argparse import Namespace, ArgumentParser

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


def arg_parse() -> Namespace:
    """Parse command line arguments"""
    from configs.fontdiffuser import get_parser

    parser: ArgumentParser = get_parser()
    
    # Original arguments
    parser.add_argument("--ckpt_dir", type=str, default=None)
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--controlnet", type=bool, default=False, 
                        help="If in demo mode, the controlnet can be added.")
    parser.add_argument("--character_input", action="store_true")
    parser.add_argument("--content_character", type=str, default=None,
                        help="Single character, comma-separated list, or path to txt file")
    parser.add_argument("--characters_file", type=str, default=None,
                        help="Path to text file with one character per line")
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
    
    args: Namespace = parser.parse_args()
    
    style_image_size: int = getattr(args, "style_image_size", 96)
    content_image_size: int = getattr(args, "content_image_size", 96)
    args.style_image_size = (style_image_size, style_image_size)
    args.content_image_size = (content_image_size, content_image_size)

    return args


class FontManager:
    """Manages single or multiple font files"""
    
    def __init__(self, ttf_path: str) -> None:
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
                'path': ttf_path,
                'font': None,  # Lazy load
                'name': font_name
            }
            print(f"✓ Font loaded: {font_name}")
            
        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions: set = {'.ttf', '.otf', '.TTF', '.OTF'}
            font_files: List[str] = [
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
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
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
    def get_font(self, font_name: str) -> Any:
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
        font_path: str = self.get_font_path(font_name)
        return is_char_in_font(font_path=font_path, char=char)
    
    def get_available_chars_for_font(self, font_name: str, 
                                     characters: List[str]) -> List[str]:
        """Get list of characters available in specific font"""
        return [
            char for char in characters 
            if self.is_char_in_font(font_name, char)
        ]


def parse_characters(content_character: Optional[str] = None, 
                    characters_file: Optional[str] = None) -> List[str]:
    """
    Parse character input from multiple sources
    
    Args:
        content_character: Single character, comma-separated list, or path to txt file
        characters_file: Path to text file with one character per line
    
    Returns:
        List of individual characters
    """
    chars: List[str] = []
    
    # Priority 1: characters_file argument
    if characters_file and os.path.isfile(characters_file):
        print(f"Loading characters from file: {characters_file}")
        with open(characters_file, 'r', encoding='utf-8') as f:
            for line in f:
                line_stripped: str = line.strip()
                if line_stripped and not line_stripped.startswith('#'):  # Skip empty lines and comments
                    # Each line should be a single character
                    if len(line_stripped) == 1:
                        chars.append(line_stripped)
                    else:
                        # If line has multiple chars, treat each as separate
                        chars.extend(list(line_stripped))
        print(f"  Loaded {len(chars)} characters")
        return chars
    
    # Priority 2: content_character argument
    if content_character:
        # Check if it's a file path
        if os.path.isfile(content_character):
            print(f"Loading characters from file: {content_character}")
            with open(content_character, 'r', encoding='utf-8') as f:
                for line in f:
                    line_stripped: str = line.strip()
                    if line_stripped and not line_stripped.startswith('#'):
                        if len(line_stripped) == 1:
                            chars.append(line_stripped)
                        else:
                            chars.extend(list(line_stripped))
            print(f"  Loaded {len(chars)} characters")
            return chars
        
        # Check if comma-separated
        if ',' in content_character:
            chars = [c.strip() for c in content_character.split(',') if c.strip()]
        else:
            # Single character
            stripped: str = content_character.strip()
            chars = [stripped] if len(stripped) == 1 else list(stripped)
    
    return chars


def get_content_transform(content_image_size: Tuple[int, int]) -> transforms.Compose:
    """Cached content transform"""
    return transforms.Compose([
        transforms.Resize(content_image_size, 
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def get_style_transform(style_image_size: Tuple[int, int]) -> transforms.Compose:
    """Cached style transform"""
    return transforms.Compose([
        transforms.Resize(style_image_size,
                         interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])


def image_process_batch(
    args: Namespace, 
    characters: List[str], 
    font_manager: FontManager, 
    font_name: str, 
    style_image_path: str
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[Image.Image]], Optional[List[str]]]:
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
    style_image: Image.Image = Image.open(style_image_path).convert('RGB')
    style_transform: transforms.Compose = get_style_transform(args.style_image_size)
    
    # Get font
    font: Any = font_manager.get_font(font_name)
    content_transform: transforms.Compose = get_content_transform(args.content_image_size)
    
    # Get available characters
    available_chars: List[str] = font_manager.get_available_chars_for_font(font_name, characters)
    
    if not available_chars:
        print(f"Warning: No characters available in font '{font_name}'")
        return None, None, None, None
    
    # Generate content images
    content_images: List[torch.Tensor] = []
    content_images_pil: List[Image.Image] = []
    
    for char in available_chars:
        try:
            content_image: Image.Image = ttf2im(font=font, char=char)
            if content_image is None:
                continue
            content_images_pil.append(content_image.copy())
            content_images.append(content_transform(content_image))
        except Exception as e:
            print(f"Error processing character '{char}': {e}")
            continue
    
    if not content_images:
        return None, None, None, None
    
    # Stack into batch
    content_batch: torch.Tensor = torch.stack(content_images)
    style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(len(content_images), 1, 1, 1)
    
    return content_batch, style_batch, content_images_pil, available_chars


def image_process(
    args: Namespace, 
    content_image: Optional[Image.Image] = None, 
    style_image: Optional[Image.Image] = None, 
    content_character: Optional[str] = None, 
    font_manager: Optional[FontManager] = None, 
    font_name: Optional[str] = None
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Image.Image]]:
    """
    Original single-image processing (maintained for compatibility)
    """
    char_to_use: Optional[str] = content_character if content_character else getattr(args, "content_character", None)
    
    if not args.demo:
        # Read content image and style image
        if args.character_input:
            assert char_to_use is not None, "The content_character should not be None."
            
            if font_manager is not None and font_name is not None:
                if not font_manager.is_char_in_font(font_name, char_to_use):
                    return None, None, None
                font: Any = font_manager.get_font(font_name)
            else:
                if not is_char_in_font(font_path=args.ttf_path, char=char_to_use):
                    return None, None, None
                font: Any = load_ttf(ttf_path=args.ttf_path)
            
            content_image_: Optional[Image.Image] = ttf2im(font=font, char=char_to_use)
            if content_image_ is None:
                return None, None, None
            content_image_pil: Image.Image = content_image_.copy()
            content_image = content_image_
        else:
            if content_image is None:
                content_image = Image.open(args.content_image_path).convert('RGB')
            content_image_pil: Optional[Image.Image] = None
        
        if style_image is None:
            style_image = Image.open(args.style_image_path).convert('RGB')
    else:
        assert style_image is not None, "The style image should not be None."
        if args.character_input:
            assert char_to_use is not None, "The content_character should not be None."
            
            if font_manager is not None and font_name is not None:
                if not font_manager.is_char_in_font(font_name, char_to_use):
                    return None, None, None
                font: Any = font_manager.get_font(font_name)
            else:
                if not is_char_in_font(font_path=args.ttf_path, char=char_to_use):
                    return None, None, None
                font: Any = load_ttf(ttf_path=args.ttf_path)
            
            content_image_: Optional[Image.Image] = ttf2im(font=font, char=char_to_use)
            if content_image_ is None:
                return None, None, None
            content_image = content_image_
        else:
            assert content_image is not None, "The content image should not be None."
        content_image_pil: Optional[Image.Image] = None
        
    # Use cached transforms
    content_transform: transforms.Compose = get_content_transform(args.content_image_size)
    style_transform: transforms.Compose = get_style_transform(args.style_image_size)
    
    content_image_tensor: torch.Tensor = content_transform(content_image)[None, :]
    style_image_tensor: torch.Tensor = style_transform(style_image)[None, :]

    return content_image_tensor, style_image_tensor, content_image_pil

def load_state_dict_auto(path: str):
    """
    Load a state_dict from .pth or .safetensors file automatically.
    """
    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file as safe_load_file
        except ImportError:
            raise ImportError("Please install safetensors to load .safetensors files.")
        return safe_load_file(path)
    else:
        return torch.load(path, map_location='cpu')

def load_fontdiffuser_pipeline(args: Namespace) -> FontDiffuserDPMPipeline:
    """
    Load FontDiffuser pipeline with SAFE optimizations
    Only applies optimizations that don't change model architecture
    """
    print("Loading FontDiffuser pipeline...")
    
    # Load the model state_dict (original architecture preserved)
    unet: Any = build_unet(args=args)
    unet_ckpt_path = f"{args.ckpt_dir}/unet.safetensors" if os.path.exists(f"{args.ckpt_dir}/unet.safetensors") else f"{args.ckpt_dir}/unet.pth"
    unet.load_state_dict(load_state_dict_auto(unet_ckpt_path))

    style_encoder: Any = build_style_encoder(args=args)
    style_encoder_ckpt_path = f"{args.ckpt_dir}/style_encoder.safetensors" if os.path.exists(f"{args.ckpt_dir}/style_encoder.safetensors") else f"{args.ckpt_dir}/style_encoder.pth"
    style_encoder.load_state_dict(load_state_dict_auto(style_encoder_ckpt_path))

    content_encoder: Any = build_content_encoder(args=args)
    content_encoder_ckpt_path = f"{args.ckpt_dir}/content_encoder.safetensors" if os.path.exists(f"{args.ckpt_dir}/content_encoder.safetensors") else f"{args.ckpt_dir}/content_encoder.pth"
    content_encoder.load_state_dict(load_state_dict_auto(content_encoder_ckpt_path))

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
    model: FontDiffuserModelDPM = FontDiffuserModelDPM(
        unet=unet,
        style_encoder=style_encoder,
        content_encoder=content_encoder
    )
    
    # Move to device with proper dtype
    dtype: torch.dtype = torch.float16 if args.fp16 else torch.float32
    model.to(args.device, dtype=dtype)
    model.eval()  # Set to evaluation mode
    
    print("✓ Model moved to device")

    # Load the training ddpm_scheduler
    train_scheduler: Any = build_ddpm_scheduler(args=args)
    print("✓ Loaded training DDPM scheduler successfully")

    # Load the DPM_Solver to generate the sample
    pipe: FontDiffuserDPMPipeline = FontDiffuserDPMPipeline(
        model=model,
        ddpm_train_scheduler=train_scheduler,
        model_type=getattr(args, "model_type", None),
        guidance_type=getattr(args, "guidance_type", None),
        guidance_scale=getattr(args, "guidance_scale", 7.5),
    )
    print("✓ Loaded DPM-Solver pipeline successfully")

    return pipe


def sampling_batch(
    args: Namespace, 
    pipe: FontDiffuserDPMPipeline, 
    characters: List[str], 
    font_manager: FontManager,
    font_name: str, 
    style_image_path: str, 
    style_name: str = "style0",
    save_content_images: bool = True
) -> Tuple[Optional[List[Image.Image]], Optional[List[str]], float]:
    """
    Batch sampling for multiple characters with single font and style
    
    Args:
        args: Arguments
        pipe: Pipeline
        characters: List of characters
        font_manager: FontManager instance
        font_name: Font name
        style_image_path: Path to style image
        style_name: Name of style (e.g., "style0")
        save_content_images: Whether to save content character images
    
    Returns:
        Tuple of (images, valid_chars, inference_time)
    """
    # Process images in batch
    content_batch, style_batch, content_pils, valid_chars = image_process_batch(
        args, characters, font_manager, font_name, style_image_path
    )
    
    if content_batch is None or valid_chars is None or content_pils is None or style_batch is None:
        return None, None, 0.0
    
    # Set seed for reproducibility
    if hasattr(args, "seed") and args.seed:
        set_seed(seed=args.seed)
    
    # Save content images if requested
    if save_content_images and getattr(args, "save_image", False):
        content_dir: str = os.path.join(args.save_image_dir, "ContentImage", font_name)
        os.makedirs(content_dir, exist_ok=True)
        
        for char, pil_img in zip(valid_chars, content_pils):
            pil_img.save(os.path.join(content_dir, f"{char}.png"))
    
    with torch.no_grad():
        dtype: torch.dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32
        content_batch = content_batch.to(args.device, dtype=dtype)
        style_batch = style_batch.to(args.device, dtype=dtype)
        
        # Apply channels-last if enabled
        if getattr(args, "channels_last", False):
            content_batch = content_batch.to(memory_format=torch.channels_last)
            style_batch = style_batch.to(memory_format=torch.channels_last)
        
        print(f"  Sampling {len(valid_chars)} characters with DPM-Solver++ ...")
        start: float = time.time()
        
        # Process in batches
        all_images: List[Image.Image] = []
        batch_size: int = getattr(args, "batch_size", 1)
        
        for i in range(0, len(content_batch), batch_size):
            batch_content: torch.Tensor = content_batch[i:i+batch_size]
            batch_style: torch.Tensor = style_batch[i:i+batch_size]
            
            images: List[Image.Image] = pipe.generate(
                content_images=batch_content,
                style_images=batch_style,
                batch_size=len(batch_content),
                order=getattr(args, "order", None),
                num_inference_steps=getattr(args, "num_inference_steps", 20),
                content_encoder_downsample_size=getattr(args, "content_encoder_downsample_size", None),
                t_start=getattr(args, "t_start", None),
                t_end=getattr(args, "t_end", None),
                dm_size=getattr(args, "content_image_size", (96, 96)),
                algorithm_type=getattr(args, "algorithm_type", None),
                skip_type=getattr(args, "skip_type", None),
                method=getattr(args, "method", None),
                correcting_x0_fn=getattr(args, "correcting_x0_fn", None)
            )
            
            all_images.extend(images)
        
        end: float = time.time()
        inference_time: float = end - start
        
        # Save generated images
        if getattr(args, "save_image", False):
            target_dir: str = os.path.join(args.save_image_dir, "TargetImage", font_name, style_name)
            os.makedirs(target_dir, exist_ok=True)
            
            for char, img in zip(valid_chars, all_images):
                img_name: str = f"{style_name}+{char}.png"
                img_path: str = os.path.join(target_dir, img_name)
                # Save directly using PIL
                img.save(img_path)
        
        print(f"  ✓ Generated {len(all_images)} images in {inference_time:.2f}s ({inference_time/len(all_images):.3f}s/img)")
        
        return all_images, valid_chars, inference_time


def sampling(
    args: Namespace, 
    pipe: FontDiffuserDPMPipeline, 
    content_image: Optional[Image.Image] = None, 
    style_image: Optional[Image.Image] = None, 
    font_manager: Optional[FontManager] = None, 
    font_name: Optional[str] = None
) -> Optional[Image.Image]:
    """
    Original single-image sampling (maintained for compatibility)
    """
    if not getattr(args, "demo", False):
        os.makedirs(args.save_image_dir, exist_ok=True)
        # saving sampling config
        save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")

    if hasattr(args, "seed") and args.seed:
        set_seed(seed=args.seed)
    
    content_image_tensor, style_image_tensor, content_image_pil = image_process(
        args=args, 
        content_image=content_image, 
        style_image=style_image,
        font_manager=font_manager,
        font_name=font_name
    )
    
    if content_image_tensor is None:
        print(f"The content_character you provided is not in the ttf. "
              f"Please change the content_character or you can change the ttf.")
        return None

    with torch.no_grad():
        dtype: torch.dtype = torch.float16 if getattr(args, "fp16", False) else torch.float32
        content_image_tensor = content_image_tensor.to(args.device, dtype=dtype)
        style_image_tensor = style_image_tensor.to(args.device, dtype=dtype)
        
        # Apply channels-last if enabled
        if getattr(args, "channels_last", False):
            content_image_tensor = content_image_tensor.to(memory_format=torch.channels_last)
            style_image_tensor = style_image_tensor.to(memory_format=torch.channels_last)
        
        print(f"Sampling by DPM-Solver++ ......")
        start: float = time.time()
        images: List[Image.Image] = pipe.generate(
            content_images=content_image_tensor,
            style_images=style_image_tensor,
            batch_size=1,
            order=getattr(args, "order", None),
            num_inference_steps=getattr(args, "num_inference_steps", 20),
            content_encoder_downsample_size=getattr(args, "content_encoder_downsample_size", None),
            t_start=getattr(args, "t_start", None),
            t_end=getattr(args, "t_end", None),
            dm_size=getattr(args, "content_image_size", (96, 96)),
            algorithm_type=getattr(args, "algorithm_type", None),
            skip_type=getattr(args, "skip_type", None),
            method=getattr(args, "method", None),
            correcting_x0_fn=getattr(args, "correcting_x0_fn", None)
        )
        end: float = time.time()

        if getattr(args, "save_image", False):
            print(f"Saving the image ......")
            save_single_image(save_dir=args.save_image_dir, image=images[0])
            if getattr(args, "character_input", False):
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=content_image_pil,
                                            content_image_path=None,
                                            style_image_path=args.style_image_path,
                                            resolution=getattr(args, "resolution", 96))
            else:
                save_image_with_content_style(save_dir=args.save_image_dir,
                                            image=images[0],
                                            content_image_pil=None,
                                            content_image_path=args.content_image_path,
                                            style_image_path=args.style_image_path,
                                            resolution=getattr(args, "resolution", 96))
            print(f"Finish the sampling process, costing time {end - start}s")
        
        return images[0]


def main() -> None:
    """Main function"""
    args: Namespace = arg_parse()
    
    print("\n" + "="*60)
    print("FONTDIFFUSER - OPTIMIZED SAMPLING")
    print("="*60)
    print(f"Model: {args.ckpt_dir}")
    print(f"Device: {args.device}")
    print(f"FP16: {getattr(args, 'fp16', False)}")
    print(f"Channels Last: {getattr(args, 'channels_last', False)}")
    print(f"Batch Size: {getattr(args, 'batch_size', 1)}")
    print("="*60 + "\n")
    
    # Load pipeline
    pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(args=args)
    
    # Parse characters from file or argument
    characters: List[str] = parse_characters(getattr(args, "content_character", None), getattr(args, "characters_file", None))
    
    # Check if multi-character or multi-font mode
    if getattr(args, "character_input", False) and characters:
        
        if len(characters) > 1 or os.path.isdir(args.ttf_path):
            # Multi-character or multi-font mode
            print(f"\n{'='*60}")
            print("BATCH MODE ACTIVATED")
            print(f"Characters: {len(characters)} - {characters[:10]}{'...' if len(characters) > 10 else ''}")
            print('='*60)
            
            # Load font manager
            font_manager: FontManager = FontManager(args.ttf_path)
            font_names: List[str] = font_manager.get_font_names()
            
            if not getattr(args, "demo", False):
                os.makedirs(args.save_image_dir, exist_ok=True)
                save_args_to_yaml(args=args, output_file=f"{args.save_image_dir}/sampling_config.yaml")
            
            # Determine style name from path
            style_name: str = os.path.splitext(os.path.basename(args.style_image_path))[0]
            
            # Process each font
            total_generated: int = 0
            for font_idx, font_name in enumerate(font_names):
                print(f"\n{'='*60}")
                print(f"[Font {font_idx+1}/{len(font_names)}] {font_name}")
                print('='*60)
                
                # Get available characters
                available: List[str] = font_manager.get_available_chars_for_font(font_name, characters)
                print(f"  Available characters: {len(available)}/{len(characters)}")
                
                if not available:
                    print("  ⚠ Skipping font (no characters available)")
                    continue
                
                # Sample in batch
                images, valid_chars, inf_time = sampling_batch(
                    args, pipe, characters, font_manager, 
                    font_name, args.style_image_path, style_name,
                    save_content_images=True
                )
                
                if images is None:
                    continue
                
                total_generated += len(images)
            
            print("\n" + "="*60)
            print("✓ BATCH PROCESSING COMPLETE")
            print("="*60)
            print(f"Total images generated: {total_generated}")
            print(f"\nOutput structure:")
            print(f"  {args.save_image_dir}/")
            print(f"    ├── ContentImage/")
            for font in font_manager.get_font_names():
                print(f"        ├── {font}/")
                print(f"        │   └── {style_name}/")
                print(f"        │       └── [generated images]")
            print("="*60)
            
        else:
            # Single character mode
            out_image: Optional[Image.Image] = sampling(args=args, pipe=pipe)
    else:
        # Original single-image mode
        out_image: Optional[Image.Image] = sampling(args=args, pipe=pipe)


if __name__=="__main__":
    main()