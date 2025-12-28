"""
Batch sampling and evaluation for FontDiffuser
Generates images in FontDiffuser standard training format:
data_examples/train/ContentImage/ + TargetImage.png/styleX/
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Import evaluation metrics
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    print("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE = False

try:
    from pytorch_fid import fid_score
    FID_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-fid not available. Install with: pip install pytorch-fid")
    FID_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: scikit-image not available. Install with: pip install scikit-image")
    SSIM_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE = False

# Import FontDiffuser modules
from sample_optimized import (
    load_fontdiffuser_pipeline_safe,
    get_content_transform,
    get_style_transform
)
from utils import load_ttf, ttf2im, is_char_in_font


class FontManager:
    """Manages multiple font files"""
    
    def __init__(self, ttf_path: str):
        """
        Initialize font manager
        
        Args:
            ttf_path: Path to a single font file or directory containing fonts
        """
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
                'font': load_ttf(ttf_path),
                'name': font_name
            }
            print(f"✓ Loaded font: {font_name}")
            
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
                try:
                    self.fonts[font_name] = {
                        'path': font_path,
                        'font': load_ttf(font_path),
                        'name': font_name
                    }
                    print(f"✓ Loaded: {font_name}")
                except Exception as e:
                    print(f"✗ Failed to load {font_name}: {e}")
            
            print('='*60)
            print(f"Successfully loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")
    
    def get_font_names(self) -> List[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())
    
    def get_font(self, font_name: str):
        """Get font object by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]['font']
    
    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]['path']
    
    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font"""
        font_path = self.get_font_path(font_name)
        return is_char_in_font(font_path, char)
    
    def get_available_chars_for_font(self, font_name: str, 
                                     characters: List[str]) -> List[str]:
        """Get list of characters available in specific font"""
        return [
            char for char in characters 
            if self.is_char_in_font(font_name, char)
        ]


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID"""
    
    def __init__(self, device='cuda:0'):
        self.device = device
        
        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn = lpips.LPIPS(net='alex').to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
        
        self.transform_to_tensor = transforms.ToTensor()
    
    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0
        
        # Convert to tensors [-1, 1]
        img1_tensor = self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
        img2_tensor = self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
        
        with torch.no_grad():
            lpips_value = self.lpips_fn(img1_tensor, img2_tensor).item()
        
        return lpips_value
    
    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0
        
        # Convert to grayscale numpy arrays
        img1_gray = np.array(img1.convert('L'))
        img2_gray = np.array(img2.convert('L'))
        
        ssim_value = ssim(img1_gray, img2_gray, data_range=255)
        return ssim_value
    
    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images"""
        if not FID_AVAILABLE:
            return -1.0
        
        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir],
                batch_size=50,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"Error computing FID: {e}")
            return -1.0
    
    def save_image(self, image: Image.Image, path: str):
        """Save PIL image to path"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Batch sampling and evaluation')
    
    # Input/Output
    parser.add_argument('--characters', type=str, required=True,
                       help='Comma-separated list of characters or path to text file')
    parser.add_argument('--style_images', type=str, required=True,
                       help='Comma-separated paths to style images or directory')
    parser.add_argument('--output_dir', type=str, default='data_examples/train',
                       help='Output directory (will create ContentImage/ and TargetImage.png/ subdirs)')
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                       help='Directory with ground truth images for evaluation')
    
    # Model configuration
    parser.add_argument('--ckpt_dir', type=str, required=True,
                       help='Checkpoint directory')
    parser.add_argument('--ttf_path', type=str, required=True,
                       help='Path to TTF font file or directory with multiple fonts')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    
    # Generation parameters
    parser.add_argument('--num_inference_steps', type=int, default=15,
                       help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='Guidance scale')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for generation')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Optimization flags
    parser.add_argument('--fp16', action='store_true', default=True,
                       help='Use FP16 precision')
    parser.add_argument('--compile', action='store_true', default=False,
                       help='Use torch.compile')
    parser.add_argument('--channels_last', action='store_true', default=True,
                       help='Use channels last memory format')
    parser.add_argument('--enable_xformers', action='store_true', default=False,
                       help='Enable xformers')
    parser.add_argument('--fast_sampling', action='store_true', default=True,
                       help='Use fast sampling mode')
    
    # Evaluation flags
    parser.add_argument('--evaluate', action='store_true', default=True,
                       help='Evaluate generated images')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                       help='Compute FID (requires ground truth)')
    parser.add_argument('--enable_attention_slicing', action='store_true', default=False,
                       help='Enable attention slicing for memory efficiency')
    
    # Wandb configuration
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Log results to Weights & Biases')
    parser.add_argument('--wandb_project', type=str, default='fontdiffuser-eval',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
    return parser.parse_args()


def load_characters(characters_arg: str) -> List[str]:
    """Load characters from file or comma-separated string"""
    chars = []
    if os.path.isfile(characters_arg):
        with open(characters_arg, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                char = line.strip()
                if not char:
                    continue
                if len(char) != 1:
                    print(f"Warning: Skipping line {i}: expected 1 char, got {len(char)}: '{char}'")
                    continue
                chars.append(char)
    else:
        for c in [x.strip() for x in characters_arg.split(',') if x.strip()]:
            if len(c) != 1:
                raise ValueError(f"Invalid character in argument: '{c}' (must be single char)")
            chars.append(c)
    
    print(f"Successfully loaded {len(chars)} single characters.")
    return chars


def load_style_images(style_images_arg: str) -> List[str]:
    """Load style image paths from comma-separated string or directory"""
    if os.path.isdir(style_images_arg):
        # Load all images from directory
        image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        style_paths = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()
    else:
        style_paths = [p.strip() for p in style_images_arg.split(',')]
    
    return style_paths


def create_args_namespace(args):
    """Create args namespace for FontDiffuser pipeline"""
    from argparse import Namespace
    
    try:
        from configs.fontdiffuser import get_parser
        parser = get_parser()
        default_args = parser.parse_args([])
    except Exception:
        default_args = Namespace()
    
    # Override with user arguments
    for key, value in vars(args).items():
        setattr(default_args, key, value)
    
    # Ensure image sizes are tuples
    if not hasattr(default_args, 'style_image_size'):
        default_args.style_image_size = (96, 96)
    elif isinstance(default_args.style_image_size, int):
        default_args.style_image_size = (default_args.style_image_size, default_args.style_image_size)
    
    if not hasattr(default_args, 'content_image_size'):
        default_args.content_image_size = (96, 96)
    elif isinstance(default_args.content_image_size, int):
        default_args.content_image_size = (default_args.content_image_size, default_args.content_image_size)
    
    # Set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    default_args.resolution = 96
    
    # Generation parameters (ensure they exist)
    default_args.algorithm_type = getattr(default_args, 'algorithm_type', 'dpmsolver++')
    default_args.guidance_type = getattr(default_args, 'guidance_type', 'classifier-free')
    default_args.method = getattr(default_args, 'method', 'multistep')
    default_args.order = getattr(default_args, 'order', 2)
    default_args.model_type = getattr(default_args, 'model_type', 'noise')
    default_args.t_start = getattr(default_args, 't_start', 1.0)
    default_args.t_end = getattr(default_args, 't_end', 1e-3)
    default_args.skip_type = getattr(default_args, 'skip_type', 'time_uniform')
    default_args.correcting_x0_fn = getattr(default_args, 'correcting_x0_fn', None)
    default_args.content_encoder_downsample_size = getattr(default_args, 'content_encoder_downsample_size', 3)
    
    return default_args


def generate_content_images(characters: List[str], font_manager: FontManager,
                           output_dir: str, args) -> Dict[str, str]:
    """
    Generate and save content character images (single set, using first available font)
    Output: data_examples/train/ContentImage/charX.png
    
    Returns:
        Dict mapping char -> path
    """
    content_dir = os.path.join(output_dir, 'ContentImage')
    os.makedirs(content_dir, exist_ok=True)
    
    # Use first available font
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")
    
    font_name = font_names[0]
    font = font_manager.get_font(font_name)
    
    print(f"\n{'='*60}")
    print(f"Generating Content Images")
    print(f"Using font: {font_name}")
    print(f"Characters: {len(characters)}")
    print('='*60)
    
    char_paths = {}
    
    for idx, char in enumerate(tqdm(characters, desc="Generating content images")):
        if not font_manager.is_char_in_font(font_name, char):
            print(f"  Warning: '{char}' not in font, skipping...")
            continue
        
        try:
            content_img = ttf2im(font=font, char=char)
            char_path = os.path.join(content_dir, f'char{idx}.png')
            content_img.save(char_path)
            char_paths[char] = char_path
        except Exception as e:
            print(f"  ✗ Error generating '{char}': {e}")
    
    print(f"✓ Generated {len(char_paths)} content images")
    print('='*60)
    
    return char_paths


def sampling_batch_optimized(args, pipe, characters: List[str], 
                             style_image_path: str, font_manager: FontManager,
                             font_name: str):
    """Batch sampling for multiple characters with specific font"""
    
    # Get available characters for this font
    available_chars = font_manager.get_available_chars_for_font(font_name, characters)
    
    if not available_chars:
        return None, None, None
    
    # Load style image
    style_image = Image.open(style_image_path).convert('RGB')
    style_transform = get_style_transform(args)
    
    font = font_manager.get_font(font_name)
    content_transform = get_content_transform(args)
    
    # Generate content images
    content_images = []
    content_images_pil = []
    
    for char in available_chars:
        try:
            content_image = ttf2im(font=font, char=char)
            content_images_pil.append(content_image.copy())
            content_images.append(content_transform(content_image))
        except Exception as e:
            print(f"    ✗ Error processing '{char}': {e}")
            continue
    
    if not content_images:
        return None, None, None
    
    # Stack into batch
    content_batch = torch.stack(content_images)
    style_batch = style_transform(style_image)[None, :].repeat(len(content_images), 1, 1, 1)
    
    with torch.no_grad():
        dtype = torch.float16 if args.fp16 else torch.float32
        content_batch = content_batch.to(args.device, dtype=dtype)
        style_batch = style_batch.to(args.device, dtype=dtype)
        
        start = time.perf_counter()
        
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
                correcting_x0_fn=args.correcting_x0_fn)
            
            all_images.extend(images)
        
        end = time.perf_counter()
        total_time = end - start
        
        return all_images, available_chars, total_time


def batch_generate_images(pipe, characters: List[str], style_paths: List[str],
                          output_dir: str, args, evaluator: QualityEvaluator,
                          font_manager: FontManager):
    """
    Generate images in batches for all fonts and styles
    Output: data_examples/train/TargetImage.png/styleX/styleX+charY.png
    """
    
    results = {
        'generations': [],
        'metrics': {
            'lpips': [],
            'ssim': [],
            'inference_times': []
        },
        'fonts': font_manager.get_font_names(),
        'total_chars': len(characters),
        'total_styles': len(style_paths)
    }
    
    # Create TargetImage.png directory
    target_base_dir = os.path.join(output_dir, 'TargetImage.png')
    os.makedirs(target_base_dir, exist_ok=True)
    
    font_names = font_manager.get_font_names()
    
    print(f"\n{'='*60}")
    print(f"BATCH GENERATION")
    print('='*60)
    print(f"Fonts: {len(font_names)}")
    print(f"Styles: {len(style_paths)}")
    print(f"Characters: {len(characters)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Inference steps: {args.num_inference_steps}")
    print('='*60)
    
    # Use first font for generation (or loop through all if needed)
    font_name = font_names[0]
    print(f"\nUsing font: {font_name}")
    
    available_chars = font_manager.get_available_chars_for_font(font_name, characters)
    print(f"Available characters: {len(available_chars)}/{len(characters)}")
    
    # Create character index mapping
    char_to_idx = {char: idx for idx, char in enumerate(characters)}
    
    for style_idx, style_path in enumerate(style_paths):
        style_name = f"style{style_idx}"
        style_dir = os.path.join(target_base_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)
        
        print(f"\n[{style_idx+1}/{len(style_paths)}] Processing {style_name}")
        print(f"  Style image: {os.path.basename(style_path)}")
        
        try:
            images, valid_chars, batch_time = sampling_batch_optimized(
                args, pipe, characters, style_path, font_manager, font_name
            )
            
            if images is None:
                print(f"  ⚠ No images generated")
                continue
            
            print(f"  Generated {len(images)} images in {batch_time:.2f}s "
                  f"({batch_time/len(images):.3f}s/img)")
            
            # Save generated images
            for char, img in zip(valid_chars, images):
                char_idx = char_to_idx.get(char, 0)
                img_name = f"{style_name}+char{char_idx}.png"
                img_path = os.path.join(style_dir, img_name)
                evaluator.save_image(img, img_path)
                
                # Store generation info
                results['generations'].append({
                    'character': char,
                    'char_index': char_idx,
                    'font': font_name,
                    'style': style_name,
                    'style_path': style_path,
                    'output_path': img_path
                })
            
            # Store timing
            results['metrics']['inference_times'].append({
                'style': style_name,
                'total_time': batch_time,
                'num_images': len(images),
                'time_per_image': batch_time / len(images) if images else 0
            })
            
            print(f"  ✓ Saved {len(images)} images to {style_dir}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print(f"✓ Generation complete! Total images: {len(results['generations'])}")
    print("="*60)
    
    return results


def evaluate_results(results: Dict, evaluator: QualityEvaluator, 
                     ground_truth_dir: str = None, compute_fid: bool = False):
    """Evaluate generated images against ground truth"""
    
    if not results['generations']:
        print("No images to evaluate")
        return results
    
    print("\n" + "="*60)
    print("EVALUATING IMAGE QUALITY")
    print("="*60)
    
    lpips_scores = []
    ssim_scores = []
    
    if ground_truth_dir and os.path.isdir(ground_truth_dir):
        print(f"\nComputing LPIPS and SSIM against ground truth...")
        
        for gen_info in tqdm(results['generations']):
            char = gen_info['character']
            gen_path = gen_info['output_path']
            
            # Find corresponding ground truth
            gt_pattern = f"*{char}*.png"
            gt_files = list(Path(ground_truth_dir).glob(gt_pattern))
            
            if not gt_files:
                continue
            
            gt_path = str(gt_files[0])
            
            try:
                gen_img = Image.open(gen_path).convert('RGB')
                gt_img = Image.open(gt_path).convert('RGB')
                
                # Resize to same size if needed
                if gen_img.size != gt_img.size:
                    gt_img = gt_img.resize(gen_img.size, Image.BILINEAR)
                
                lpips_val = evaluator.compute_lpips(gen_img, gt_img)
                ssim_val = evaluator.compute_ssim(gen_img, gt_img)
                
                if lpips_val >= 0:
                    lpips_scores.append(lpips_val)
                if ssim_val >= 0:
                    ssim_scores.append(ssim_val)
                
            except Exception as e:
                print(f"Error evaluating {char}: {e}")
    
    # Store overall metrics
    if lpips_scores:
        results['metrics']['lpips'] = {
            'mean': float(np.mean(lpips_scores)),
            'std': float(np.std(lpips_scores)),
            'min': float(np.min(lpips_scores)),
            'max': float(np.max(lpips_scores))
        }
    
    if ssim_scores:
        results['metrics']['ssim'] = {
            'mean': float(np.mean(ssim_scores)),
            'std': float(np.std(ssim_scores)),
            'min': float(np.min(ssim_scores)),
            'max': float(np.max(ssim_scores))
        }
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    if lpips_scores:
        print(f"\nLPIPS: {results['metrics']['lpips']['mean']:.4f} ± {results['metrics']['lpips']['std']:.4f}")
    
    if ssim_scores:
        print(f"SSIM:  {results['metrics']['ssim']['mean']:.4f} ± {results['metrics']['ssim']['std']:.4f}")
    
    print("="*60)
    
    return results


def log_to_wandb(results: Dict, args):
    """Log results to Weights & Biases"""
    if not WANDB_AVAILABLE or not args.use_wandb:
        return
    
    run_name = args.wandb_run_name or f"fontdiffuser_{time.strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            'num_characters': results.get('total_chars', 0),
            'num_styles': results.get('total_styles', 0),
            'total_generations': len(results['generations']),
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
            'batch_size': args.batch_size,
            'fp16': args.fp16,
        }
    )
    
    # Log overall metrics
    if 'lpips' in results['metrics'] and isinstance(results['metrics']['lpips'], dict):
        wandb.log({
            'lpips_mean': results['metrics']['lpips']['mean'],
            'lpips_std': results['metrics']['lpips']['std'],
        })
    
    if 'ssim' in results['metrics'] and isinstance(results['metrics']['ssim'], dict):
        wandb.log({
            'ssim_mean': results['metrics']['ssim']['mean'],
            'ssim_std': results['metrics']['ssim']['std'],
        })
    
    # Log inference times
    if results['metrics']['inference_times']:
        total_time = sum(t['total_time'] for t in results['metrics']['inference_times'])
        total_images = sum(t['num_images'] for t in results['metrics']['inference_times'])
        
        wandb.log({
            'total_inference_time': total_time,
            'total_images': total_images,
            'avg_time_per_image': total_time / total_images if total_images > 0 else 0
        })
    
    # Log sample images
    sample_images = []
    for gen_info in results['generations'][:20]:  # Max 20 samples
        if os.path.exists(gen_info['output_path']):
            sample_images.append(
                wandb.Image(
                    gen_info['output_path'],
                    caption=f"{gen_info['character']} - {gen_info['style']}"
                )
            )
    
    if sample_images:
        wandb.log({"sample_generations": sample_images})
    
    wandb.finish()
    print("\n✓ Results logged to W&B")


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("FONTDIFFUSER STANDARD FORMAT GENERATION")
    print("="*60)
    
    # Load characters and styles
    characters = load_characters(args.characters)
    style_paths = load_style_images(args.style_images)
    
    # Initialize font manager
    print(f"\nInitializing font manager...")
    font_manager = FontManager(args.ttf_path)
    
    print(f"\n📊 Configuration:")
    print(f"  Characters: {len(characters)}")
    print(f"  Styles: {len(style_paths)}")
    print(f"  Output: {args.output_dir}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate content images (single set)
    char_paths = generate_content_images(
        characters, font_manager, args.output_dir, args
    )
    
    # Create args namespace for pipeline
    pipeline_args = create_args_namespace(args)
    
    # Load pipeline
    print("\nLoading FontDiffuser pipeline...")
    pipe = load_fontdiffuser_pipeline_safe(pipeline_args)
    
    # Initialize evaluator
    evaluator = QualityEvaluator(device=args.device)
    
    # Generate target images
    results = batch_generate_images(
        pipe, characters, style_paths, args.output_dir,
        pipeline_args, evaluator, font_manager
    )
    
    # Evaluate if requested
    if args.evaluate and args.ground_truth_dir:
        results = evaluate_results(
            results, evaluator, args.ground_truth_dir, args.compute_fid
        )
    
    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {results_path}")
    
    # Log to wandb
    if args.use_wandb:
        log_to_wandb(results, args)
    
    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {args.output_dir}/")
    print(f"    ├── ContentImage/")
    print(f"    │   ├── char0.png")
    print(f"    │   ├── char1.png")
    print(f"    │   └── ...")
    print(f"    └── TargetImage.png/")
    print(f"        ├── style0/")
    print(f"        │   ├── style0+char0.png")
    print(f"        │   ├── style0+char1.png")
    print(f"        │   └── ...")
    print(f"        ├── style1/")
    print(f"        │   └── ...")
    print(f"        └── ...")


if __name__ == "__main__":
    main()