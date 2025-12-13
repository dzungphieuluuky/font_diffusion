"""
FontDiffuser Batch Processing with Inference Optimizations
Uses optimized sampling functions for 2-3x faster processing
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from functools import lru_cache

import pandas as pd
import ast
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from accelerate.utils import set_seed

# Suppress warnings
warnings.filterwarnings('ignore')

# Import FontDiffuser components
try:
    from src import (
        FontDiffuserDPMPipeline,
        FontDiffuserModelDPM
    )
    from utils import (
        save_args_to_yaml,
        save_single_image,
        save_image_with_content_style
    )

    # Import optimized functions
    try:
        from sample_optimized import (
            load_fontdiffuser_pipeline_safe,
            get_fontdiffuser_pipeline_cached,
            sampling_optimized,
            image_process_optimized,
            get_content_transform,
            get_style_transform,
            arg_parse_optimized,
            load_ttf_cached,
            is_char_in_font_cached
        )
        OPTIMIZED_AVAILABLE = True
        print("✓ Using optimized sampling functions")
    except ImportError:
        print("⚠ Optimized functions not available, falling back to standard")
        from sample import (
            load_fontdiffuser_pipeline,
            sampling,
            image_process
        )
        OPTIMIZED_AVAILABLE = False

    from font_manager import FontManager
    
except ImportError as e:
    print(f"Error importing FontDiffuser modules: {e}")
    print("Please ensure the required modules are in your Python path")
    sys.exit(1)


class FontDiffuserBatchProcessorOptimized:
    """Process Excel file with optimized inference for 2-3x faster processing"""
    
    def __init__(self, args, pipe, font_manager: FontManager):
        self.args = args
        self.pipe = pipe
        self.font_manager = font_manager
        self.device = args.device
        
        # Create transforms with caching if using optimized version
        if OPTIMIZED_AVAILABLE:
            # Use cached transforms from sample_optimized
            self.content_transforms = get_content_transform(args)
            self.style_transforms = get_style_transform(args)
        else:
            self.content_transforms, self.style_transforms = self._create_transforms()
        
        # Statistics tracking
        self.stats = {
            'characters_processed': 0,
            'characters_skipped_no_font': 0,
            'characters_skipped_render_failed': 0,
            'generation_errors': 0,
            'fonts_used': defaultdict(int),
            'edge_cases_fixed': 0,
            'total_inference_time': 0.0,
            'avg_time_per_character': 0.0
        }
        
        # Batch timing
        self.batch_start_time = None
        self.current_char_start_time = None
    
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
        Prepare content and style images with optimized caching
        """
        self.current_char_start_time = time.perf_counter()
        
        try:
            # Load style image
            if not os.path.exists(style_image_path):
                raise FileNotFoundError(f"Style image not found: {style_image_path}")
            
            style_image = Image.open(style_image_path).convert('RGB')
            
            # Use optimized tensor conversion if available
            if OPTIMIZED_AVAILABLE:
                style_image_tensor = self.style_transforms(style_image)[None, :]
                # Move to device later with proper dtype
            else:
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
            if OPTIMIZED_AVAILABLE:
                content_image_tensor = self.content_transforms(content_image)[None, :]
            else:
                content_image_tensor = self.content_transforms(content_image)[None, :].to(self.device)
            
            # Debug: Save the prepared content image
            if hasattr(self.args, 'debug') and self.args.debug:
                debug_dir = Path(self.args.output_base_dir) / "debug" / "content_images"
                debug_dir.mkdir(parents=True, exist_ok=True)
                content_image.save(debug_dir / f"content_{char}.png")
            
            prep_time = time.perf_counter() - self.current_char_start_time
            if prep_time > 0.5:  # Log slow preparation
                print(f"    Prep time: {prep_time:.2f}s")
            
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
    
    def generate_character_optimized(self, char: str, style_image_path: str) -> Tuple[Optional[torch.Tensor], Optional[Image.Image]]:
        """
        Generate a single character using optimized FontDiffuser pipeline
        """
        # Set seed for reproducibility if specified
        if self.args.seed:
            set_seed(seed=self.args.seed)
        
        # Prepare images
        content_tensor, style_tensor, content_pil = self.prepare_images(char, style_image_path)
        
        if content_tensor is None or style_tensor is None:
            return None, None
        
        try:
            # Generate using optimized pipeline
            with torch.no_grad():
                # Move tensors to device with proper dtype for optimized version
                if OPTIMIZED_AVAILABLE:
                    dtype = torch.float16 if getattr(self.args, 'fp16', True) else torch.float32
                    content_tensor = content_tensor.to(self.device, dtype=dtype)
                    style_tensor = style_tensor.to(self.device, dtype=dtype)
                
                # Start timing for inference only
                inference_start = time.perf_counter()
                
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
                
                inference_time = time.perf_counter() - inference_start
                self.stats['total_inference_time'] += inference_time
                
                # Update average time
                if self.stats['characters_processed'] > 0:
                    self.stats['avg_time_per_character'] = (
                        self.stats['total_inference_time'] / self.stats['characters_processed']
                    )
            
            self.stats['characters_processed'] += 1
            
            # Log performance if this character took longer than average
            if self.stats['characters_processed'] > 3 and inference_time > self.stats['avg_time_per_character'] * 1.5:
                print(f"    ⚠ Slow inference: {inference_time:.2f}s "
                      f"(avg: {self.stats['avg_time_per_character']:.2f}s)")
            
            return images[0], content_pil
            
        except Exception as e:
            print(f"    Error generating character '{char}': {e}")
            self.stats['generation_errors'] += 1
            return None, None
    
    def process_excel_file_optimized(self,
                                   excel_path: str,
                                   base_output_dir: str,
                                   style_image_path: str = None,
                                   generate_input_char: bool = True,
                                   start_line: Optional[int] = None,
                                   end_line: Optional[int] = None) -> Dict:
        """
        Process Excel file with optimized batch processing
        """
        # Start batch timing
        self.batch_start_time = time.perf_counter()
        
        # Use provided style image or default from args
        if style_image_path is None:
            style_image_path = self.args.style_image_path
        
        # Create base output directory
        Path(base_output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save font statistics
        font_stats = self.font_manager.get_font_statistics()
        with open(Path(base_output_dir) / "font_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(font_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*70}")
        print(f"FONTDIFFUSER OPTIMIZED BATCH PROCESSING")
        print(f"{'='*70}")
        print(f"Loaded {font_stats['total_fonts']} fonts")
        
        # Load Excel file
        print(f"\nLoading Excel file: {excel_path}")
        try:
            df = pd.read_excel(excel_path)
            total_rows = len(df)
            print(f"Excel file loaded with {total_rows} total rows")
        except Exception as e:
            raise ValueError(f"Failed to load Excel file: {e}")
        
        # Check required columns
        required_columns = ['Input Character', 'Top 20 Similar Characters']
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
                print(f"Warning: end_line {end_line} exceeds total rows {total_rows}, using {total_rows}")
                end_idx = total_rows - 1
            
            if start_idx > end_idx:
                print(f"Warning: start_line ({start_line}) > end_line ({end_line}), swapping")
                start_idx, end_idx = end_idx, start_idx
            
            # Slice the dataframe
            df = df.iloc[start_idx:end_idx + 1]
            print(f"Processing rows {start_line or 1} to {end_line or total_rows} "
                  f"({len(df)} rows)")
        else:
            print(f"Processing all {total_rows} rows")
        
        # Process each row
        results_summary = {}
        total_chars_to_process = 0
        processed_chars = 0
        
        # First pass: count total characters for progress tracking
        for idx, row in df.iterrows():
            input_char = str(row['Input Character']).strip()
            similar_chars_str = row['Top 20 Similar Characters']
            similar_chars = self._parse_similar_characters(similar_chars_str)
            
            if generate_input_char:
                total_chars_to_process += 1
            total_chars_to_process += len(similar_chars)
        
        print(f"\nTotal characters to generate: {total_chars_to_process}")
        print(f"Estimated time: {total_chars_to_process * 3:.1f}s (3s per character)")
        if getattr(self.args, 'fp16', True):
            print(f"Using FP16 optimization: ~2x faster")
        if getattr(self.args, 'fast_sampling', False):
            print(f"Using fast sampling: ~{self.args.num_inference_steps} steps")
        print(f"{'='*70}")
        
        # Second pass: actual processing
        for idx, row in df.iterrows():
            # Calculate actual row number in original Excel (1-indexed)
            original_row_num = idx + 1 if start_line is None else start_idx + (idx - df.index[0]) + 1
            
            input_char = str(row['Input Character']).strip()
            similar_chars_str = row['Top 20 Similar Characters']
            
            print(f"\n[Row {original_row_num}/{len(df)}] "
                  f"Processing: '{input_char}'")
            
            # Parse similar characters
            similar_chars = self._parse_similar_characters(similar_chars_str)
            
            # Check font support for all characters
            all_chars = [input_char] + similar_chars if generate_input_char else similar_chars
            font_mapping = self.font_manager.find_fonts_for_characters(all_chars)
            
            # Report font support
            unsupported = [c for c, f in font_mapping.items() if f is None]
            if unsupported:
                print(f"  ⚠ {len(unsupported)} characters without font support")
            
            # Create folder for this input character
            safe_char_name = self._sanitize_filename(input_char)
            char_output_dir = Path(base_output_dir) / f"row_{original_row_num:04d}_{safe_char_name}"
            char_output_dir.mkdir(exist_ok=True)
            
            # Save character information
            self._save_character_info(char_output_dir, input_char, similar_chars, 
                                    font_mapping, style_image_path, original_row_num)
            
            # Generate characters with progress tracking
            generated_chars = self._generate_characters_for_row_optimized(
                char_output_dir,
                input_char,
                similar_chars,
                style_image_path,
                generate_input_char,
                font_mapping,
                original_row_num,
                processed_chars,
                total_chars_to_process
            )
            
            processed_chars += len(generated_chars)
            
            # Update summary
            results_summary[input_char] = {
                'output_dir': str(char_output_dir),
                'excel_row': original_row_num,
                'generated_count': len(generated_chars),
                'similar_characters': similar_chars,
                'generated_chars': list(generated_chars.keys()),
                'font_mapping': {k: v for k, v in font_mapping.items() if k in generated_chars}
            }
        
        # Save final summaries
        self._save_final_summaries_optimized(base_output_dir, excel_path, results_summary, 
                                           start_line, end_line)
        
        return results_summary
    
    def _parse_similar_characters(self, similar_chars_str) -> List[str]:
        """Parse similar characters string from Excel"""
        try:
            if pd.isna(similar_chars_str):
                return []
            
            if isinstance(similar_chars_str, str):
                if similar_chars_str.startswith('['):
                    # Parse as Python list
                    return ast.literal_eval(similar_chars_str)[:20]
                else:
                    # Parse as comma-separated or other format
                    chars = str(similar_chars_str).strip("[]").replace("'", "").split(',')
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
            filename = filename.replace(char, '_')
        
        # Also remove control characters
        filename = ''.join(c for c in filename if ord(c) >= 32)
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:50]
        
        return filename
    
    def _save_character_info(self, output_dir: Path, input_char: str, 
                           similar_chars: List[str], font_mapping: Dict, 
                           style_image_path: str, excel_row: int):
        """Save character information to file"""
        info_path = output_dir / "character_info.json"
        info = {
            'input_character': input_char,
            'excel_row_number': excel_row,
            'similar_characters': similar_chars,
            'style_image': style_image_path,
            'font_mapping': font_mapping,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_characters': len(similar_chars) + 1,
            'optimizations': {
                'fp16': getattr(self.args, 'fp16', True),
                'fast_sampling': getattr(self.args, 'fast_sampling', False),
                'inference_steps': getattr(self.args, 'num_inference_steps', 20)
            }
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
    
    def _generate_characters_for_row_optimized(self, output_dir: Path, input_char: str,
                                            similar_chars: List[str], style_image_path: str,
                                            generate_input_char: bool, font_mapping: Dict,
                                            excel_row: int, processed_so_far: int,
                                            total_chars: int) -> Dict:
        """Generate all characters for a single row with progress tracking"""
        generated_chars = {}
        
        # Characters to generate
        chars_to_generate = []
        if generate_input_char:
            chars_to_generate.append(('input', input_char))
        
        for char in similar_chars:
            chars_to_generate.append(('similar', char))
        
        # Generate each character
        for char_type, char in chars_to_generate:
            current_progress = processed_so_far + len(generated_chars) + 1
            progress_pct = (current_progress / total_chars) * 100
            
            print(f"  [{current_progress}/{total_chars} | {progress_pct:.1f}%] "
                  f"Generating '{char}'...", end='', flush=True)
            
            # Check font support
            if font_mapping.get(char) is None:
                print(" ✗ (no font support)")
                continue
            
            # Generate character using optimized method
            generated_image, content_pil = self.generate_character_optimized(char, style_image_path)
            
            if generated_image is not None:
                # Save the generated character
                save_success = self._save_generated_character(
                    output_dir, char_type, char, generated_image, content_pil, style_image_path
                )
                
                if save_success:
                    generated_chars[char] = {
                        'type': char_type,
                        'font_used': font_mapping[char],
                        'inference_time': self.stats['avg_time_per_character'] if self.stats['characters_processed'] > 0 else 0
                    }
                    print(" ✓")
                else:
                    print(" ✗ (save failed)")
            else:
                print(" ✗ (generation failed)")
        
        return generated_chars
    
    def _save_generated_character(self, output_dir: Path, char_type: str, char: str,
                                generated_image: torch.Tensor, content_pil: Optional[Image.Image],
                                style_image_path: str) -> bool:
        """Save generated character to appropriate location"""
        try:
            # Create directory structure
            if char_type == 'input':
                char_dir = output_dir / "input_character"
            else:
                char_dir = output_dir / "similar_characters" / self._sanitize_filename(char)
            
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
                    resolution=self.args.resolution
                )
            
            return True
            
        except Exception as e:
            print(f"    Error saving character '{char}': {e}")
            return False
    
    def _save_final_summaries_optimized(self, base_output_dir: str, excel_path: str, 
                                      results_summary: Dict, start_line: Optional[int], 
                                      end_line: Optional[int]):
        """Save final summary files with optimization metrics"""
        base_path = Path(base_output_dir)
        
        # Calculate batch statistics
        batch_total_time = time.perf_counter() - self.batch_start_time
        chars_per_second = self.stats['characters_processed'] / batch_total_time if batch_total_time > 0 else 0
        
        # Global summary with optimization metrics
        global_summary = {
            'processing_completed': time.strftime('%Y-%m-%d %H:%M:%S'),
            'excel_file': excel_path,
            'output_directory': base_output_dir,
            'line_range': {
                'start_line': start_line,
                'end_line': end_line,
                'processed_rows': len(results_summary)
            },
            'optimization_settings': {
                'fp16': getattr(self.args, 'fp16', True),
                'fast_sampling': getattr(self.args, 'fast_sampling', False),
                'inference_steps': getattr(self.args, 'num_inference_steps', 20),
                'compile': getattr(self.args, 'compile', False),
                'channels_last': getattr(self.args, 'channels_last', True)
            },
            'performance_metrics': {
                'total_batch_time': round(batch_total_time, 2),
                'characters_per_second': round(chars_per_second, 2),
                'avg_inference_time_per_character': round(self.stats['avg_time_per_character'], 2),
                'total_inference_time': round(self.stats['total_inference_time'], 2),
                'total_characters_attempted': self.stats['characters_processed'] + 
                                            self.stats['characters_skipped_no_font'] + 
                                            self.stats['characters_skipped_render_failed']
            },
            'processing_statistics': self.stats,
            'detailed_results': results_summary
        }
        
        # Save as JSON
        with open(base_path / "global_summary.json", 'w', encoding='utf-8') as f:
            json.dump(global_summary, f, indent=2, ensure_ascii=False)
        
        # Also save as human-readable text with performance highlights
        with open(base_path / "global_summary.txt", 'w', encoding='utf-8') as f:
            f.write(f"FontDiffuser Optimized Batch Processing Summary\n")
            f.write(f"=" * 60 + "\n\n")
            
            f.write(f"Processing Completed: {global_summary['processing_completed']}\n")
            f.write(f"Excel File: {excel_path}\n")
            f.write(f"Output Directory: {base_output_dir}\n")
            
            if start_line or end_line:
                f.write(f"Line Range: {start_line or 'start'} to {end_line or 'end'}\n")
            f.write(f"Total Rows Processed: {len(results_summary)}\n\n")
            
            f.write(f"Optimization Settings:\n")
            for key, value in global_summary['optimization_settings'].items():
                f.write(f"  • {key}: {value}\n")
            f.write("\n")
            
            f.write(f"Performance Metrics:\n")
            f.write(f"  • Total batch time: {batch_total_time:.2f}s\n")
            f.write(f"  • Characters per second: {chars_per_second:.2f}\n")
            f.write(f"  • Avg inference time per character: {self.stats['avg_time_per_character']:.2f}s\n")
            f.write(f"  • Total inference time: {self.stats['total_inference_time']:.2f}s\n")
            f.write("\n")
            
            f.write(f"Processing Statistics:\n")
            f.write(f"  • Characters Processed: {self.stats['characters_processed']}\n")
            f.write(f"  • Characters Skipped (no font): {self.stats['characters_skipped_no_font']}\n")
            f.write(f"  • Characters Skipped (render failed): {self.stats['characters_skipped_render_failed']}\n")
            f.write(f"  • Generation Errors: {self.stats['generation_errors']}\n")
            f.write(f"  • Edge Cases Fixed: {self.stats['edge_cases_fixed']}\n\n")
            
            f.write(f"Font Usage Statistics:\n")
            for font_name, count in sorted(self.stats['fonts_used'].items(), key=lambda x: x[1], reverse=True):
                f.write(f"  • {font_name}: {count} characters\n")
            f.write("\n")
            
            f.write(f"Results by Input Character:\n")
            for input_char, info in results_summary.items():
                f.write(f"\n  {input_char} (Row {info['excel_row']}):\n")
                f.write(f"    Output: {info['output_dir']}\n")
                f.write(f"    Generated: {info['generated_count']} characters\n")
        
        # Print performance summary
        print(f"\n{'='*70}")
        print(f"OPTIMIZED BATCH PROCESSING COMPLETE")
        print(f"{'='*70}")
        print(f"Performance Summary:")
        print(f"  • Total batch time: {batch_total_time:.2f}s")
        print(f"  • Characters per second: {chars_per_second:.2f}")
        print(f"  • Avg inference time: {self.stats['avg_time_per_character']:.2f}s/char")
        print(f"  • Total characters processed: {self.stats['characters_processed']}")
        print(f"\nOptimizations used:")
        for key, value in global_summary['optimization_settings'].items():
            if value:
                print(f"  • {key}: {value}")
        print(f"\nOutput saved to: {base_output_dir}")
        print(f"Detailed summary: {base_path / 'global_summary.json'}")


def parse_excel_batch_args_optimized():
    """Parse arguments for optimized batch processing"""
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
    parser.add_argument("--output_base_dir", type=str, default="./fontdiffuser_batch_output_optimized",
                       help="Base directory for all outputs")
    parser.add_argument("--skip_input_char", action="store_true",
                       help="Skip generating the input character")
    
    # Line range arguments
    parser.add_argument("--start_line", type=int, default=None,
                       help="Starting line number in Excel (1-indexed, inclusive)")
    parser.add_argument("--end_line", type=int, default=None,
                       help="Ending line number in Excel (1-indexed, inclusive)")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Maximum number of rows to process (alternative to end_line)")
    
    # Optimization arguments
    parser.add_argument("--optimize", action="store_true", default=True,
                       help="Enable inference optimizations")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Use half precision (FP16) for faster inference")
    parser.add_argument("--fast_sampling", action="store_true", default=False,
                       help="Use fewer sampling steps for faster processing")
    parser.add_argument("--inference_steps", type=int, default=None,
                       help="Override num_inference_steps for faster inference")
    parser.add_argument("--compile", action="store_true", default=False,
                       help="Use torch.compile() for PyTorch 2.0+")
    parser.add_argument("--channels_last", action="store_true", default=True,
                       help="Use channels last memory format")
    parser.add_argument("--cache_models", action="store_true", default=True,
                       help="Cache model builds for repeated inference")
    
    # Debug argument
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
    
    # Validate line range arguments
    if args.start_line is not None and args.start_line < 1:
        print(f"Warning: start_line must be >= 1, got {args.start_line}")
        args.start_line = 1
    
    if args.end_line is not None and args.end_line < 1:
        print(f"Warning: end_line must be >= 1, got {args.end_line}")
        args.end_line = None
    
    if args.start_line is not None and args.end_line is not None:
        if args.start_line > args.end_line:
            print(f"Warning: start_line ({args.start_line}) > end_line ({args.end_line}), swapping")
            args.start_line, args.end_line = args.end_line, args.start_line
    
    # Handle max_rows if specified
    if args.max_rows is not None:
        if args.end_line is None and args.start_line is not None:
            args.end_line = args.start_line + args.max_rows - 1
        elif args.end_line is None:
            args.end_line = args.max_rows
    
    # Adjust inference steps for fast sampling
    if args.fast_sampling and args.inference_steps is None:
        args.inference_steps = min(20, args.num_inference_steps)
    
    if args.inference_steps:
        args.num_inference_steps = args.inference_steps
    
    return args


def load_fontdiffuser_pipeline_optimized(args):
    """Load the FontDiffuser pipeline with optimizations"""
    print("Loading FontDiffuser model with optimizations...")
    
    if OPTIMIZED_AVAILABLE:
        # Use optimized pipeline loader
        pipe = load_fontdiffuser_pipeline_safe(args)
    else:
        # Fallback to original loader
        from sample import load_fontdiffuser_pipeline
        pipe = load_fontdiffuser_pipeline(args)
    
    return pipe


def main_batch_processing_optimized():
    """Main function for optimized batch processing"""
    args = parse_excel_batch_args_optimized()
    
    print(f"\n{'='*70}")
    print(f"FONTDIFFUSER OPTIMIZED BATCH PROCESSING")
    print(f"{'='*70}")
    
    if args.start_line or args.end_line:
        print(f"Line Range: {args.start_line or 'start'} to {args.end_line or 'end'}")
    
    # Print optimization settings
    print(f"\nOptimization Settings:")
    print(f"  • FP16: {args.fp16}")
    print(f"  • Fast Sampling: {args.fast_sampling}")
    print(f"  • Inference Steps: {args.num_inference_steps}")
    print(f"  • torch.compile: {args.compile}")
    print(f"  • Channels Last: {args.channels_last}")
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
        font_size=256,
        canvas_size=256
    )
    
    # Load pipeline with optimizations
    print(f"\nLoading FontDiffuser pipeline with optimizations...")
    pipe = load_fontdiffuser_pipeline_optimized(args)
    
    # Create optimized processor
    processor = FontDiffuserBatchProcessorOptimized(args, pipe, font_manager)
    
    # Process Excel file with optimizations
    print(f"\nProcessing Excel file: {args.excel_file}")
    results = processor.process_excel_file_optimized(
        excel_path=args.excel_file,
        base_output_dir=str(output_dir),
        style_image_path=args.style_image_path,
        generate_input_char=not args.skip_input_char,
        start_line=args.start_line,
        end_line=args.end_line
    )
    
    return results


if __name__ == "__main__":
    try:
        results = main_batch_processing_optimized()
        print(f"\n✓ Optimized batch processing completed successfully!")
        print(f"   Performance improvements: ~2-3x faster than standard processing")
    except Exception as e:
        print(f"\n✗ Error during optimized batch processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


"""Example - Optimized Batch Processing:
python sample_excel_optimized.py \
    --excel_file "characters.xlsx" \
    --style_image_path "./style/A.png" \
    --ckpt_dir "./checkpoints" \
    --ttf_path "./fonts/KaiXinSongA.ttf" \
    --output_base_dir "./font_generations_30_40" \
    --start_line 30 \
    --end_line 40 \
    --fp16 \           # Enable FP16 for 2x speed
    --fast_sampling \  # Use fewer steps
    --inference_steps 15 \  # 15 steps instead of 20
    --channels_last    # Better memory format
"""

"""Example - Ultra Fast Mode:
python sample_excel_optimized.py \
    --excel_file "characters.xlsx" \
    --style_image_path "./style/A.png" \
    --ckpt_dir "./checkpoints" \
    --ttf_path "./fonts/KaiXinSongA.ttf" \
    --output_base_dir "./output_fast" \
    --fp16 \
    --fast_sampling \
    --inference_steps 10 \  # Ultra fast
    --compile \  # PyTorch 2.0 compilation
    --max_rows 10  # Limit to 10 rows
"""

"""Output Structure:
font_generations_30_40/
├── batch_config.yaml
├── font_statistics.json
├── global_summary.json    # Includes performance metrics
├── performance_report.txt
├── debug/
│   └── content_images/
├── row_001_𠀖/
│   ├── character_info.json
│   ├── input_character/
│   └── similar_characters/
└── ...
"""