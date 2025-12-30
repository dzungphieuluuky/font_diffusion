"""
Create Hugging Face dataset from generated FontDiffusion images and push to Hub
Preserves and uploads original results.json metadata
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Image as HFImage
from PIL import Image as PILImage
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass
class FontDiffusionDatasetConfig:
    """Configuration for dataset creation"""
    data_dir: str  # Path to data_examples/train
    repo_id: str  # huggingface repo id (e.g., "username/fontdiffusion-dataset")
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None  # HF token for private repos
    upload_metadata: bool = True  # ✅ Upload original results.json


class FontDiffusionDatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format"""
    
    def __init__(self, config: FontDiffusionDatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.content_dir = self.data_dir / "ContentImage"
        self.target_dir = self.data_dir / "TargetImage"
        
        self._validate_structure()
    
    def _validate_structure(self) -> None:
        """Validate directory structure"""
        if not self.content_dir.exists():
            raise ValueError(f"ContentImage directory not found: {self.content_dir}")
        if not self.target_dir.exists():
            raise ValueError(f"TargetImage directory not found: {self.target_dir}")
        
        print(f"✓ Validated directory structure")
        print(f"  Content images: {self.content_dir}")
        print(f"  Target images: {self.target_dir}")
    
    def _load_results_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load results.json metadata if available
        Returns None if file doesn't exist or is corrupted
        """
        results_path = self.data_dir / "results.json"
        if not results_path.exists():
            print(f"⚠ results.json not found at {results_path}")
            return None
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ Loaded results.json")
            return metadata
        except json.JSONDecodeError as e:
            print(f"\n⚠ Warning: results.json is corrupted ({e})")
            print(f"  Proceeding without metadata...")
            return None  # ✅ Return None instead of raising
        except Exception as e:
            print(f"\n⚠ Warning: Could not load results.json ({e})")
            print(f"  Proceeding without metadata...")
            return None  # ✅ Return None instead of raising
    
    def _get_results_metadata_path(self, metadata_filename: str) -> Optional[Path]:
        """
        Get path to original metadata file if it exists
        
        Args:
            metadata_filename: Name of metadata file (e.g., "results.json" or "results_checkpoint.json")
        
        Returns:
            Path object if file exists, None otherwise
        """
        results_path = self.data_dir / metadata_filename
        if results_path.exists():
            return results_path
        return None
           
    def build_dataset(self) -> Dataset:
        """
        Build dataset with structure:
        {
            'character': str,
            'char_index': int,
            'style': str,
            'style_index': int,
            'content_image': PIL.Image,
            'target_image': PIL.Image,
            'font': str (optional)
        }
        """
        print("\n" + "="*60)
        print("BUILDING DATASET")
        print("="*60)
        
        # Load metadata (optional)
        metadata = self._load_results_metadata()
        
        dataset_rows: List[Dict[str, Any]] = []
        
        # Load all content and target images
        content_dir = self.data_dir / "ContentImage"
        target_base_dir = self.data_dir / "TargetImage"
        
        if not content_dir.exists() or not target_base_dir.exists():
            raise ValueError(f"Missing required directories in {self.data_dir}")
        
        # Build mapping from results.json
        gen_map: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (char_idx, style_idx) -> gen_info
        
        if metadata and 'generations' in metadata:
            for gen_info in metadata['generations']:
                char_idx = gen_info.get('char_index')
                style_idx = gen_info.get('style_index')
                
                if char_idx is not None and style_idx is not None:
                    gen_map[(char_idx, style_idx)] = gen_info
        
        # Iterate through target images
        for style_dir in sorted(target_base_dir.iterdir()):
            if not style_dir.is_dir():
                continue
            
            style_name = style_dir.name  # e.g., "style0"
            style_idx = int(style_name.replace('style', ''))
            
            for target_img_path in sorted(style_dir.glob("*.png")):
                # Parse filename: style0+char5.png
                filename = target_img_path.stem
                parts = filename.split('+')
                
                if len(parts) != 2:
                    continue
                
                char_idx_str = parts[1].replace('char', '')
                
                try:
                    char_idx = int(char_idx_str)
                except ValueError:
                    continue
                
                # Get content image path
                content_img_path = content_dir / f"char{char_idx}.png"
                
                if not content_img_path.exists():
                    print(f"⚠ Missing content image: {content_img_path}")
                    continue
                
                # Load images
                try:
                    content_image = PILImage.open(content_img_path).convert('RGB')
                    target_image = PILImage.open(target_img_path).convert('RGB')
                except Exception as e:
                    print(f"⚠ Error loading images for {filename}: {e}")
                    continue
                
                # Get metadata for this pair
                gen_info = gen_map.get((char_idx, style_idx), {})
                
                # Extract information - use gen_info first, then fallback
                character = gen_info.get('character', '?')
                font_name = gen_info.get('font', 'unknown')
                
                row = {
                    'character': character,
                    'char_index': char_idx,
                    'style': style_name,
                    'style_index': style_idx,
                    'content_image': content_image,
                    'target_image': target_image,
                    'font': font_name
                }
                
                dataset_rows.append(row)
        
        print(f"✓ Loaded {len(dataset_rows)} samples")
        
        if not dataset_rows:
            raise ValueError("No samples loaded!")
        
        return Dataset.from_dict({
            'character': [r['character'] for r in dataset_rows],
            'char_index': [r['char_index'] for r in dataset_rows],
            'style': [r['style'] for r in dataset_rows],
            'style_index': [r['style_index'] for r in dataset_rows],
            'content_image': [r['content_image'] for r in dataset_rows],
            'target_image': [r['target_image'] for r in dataset_rows],
            'font': [r['font'] for r in dataset_rows],
        }).cast_column('content_image', HFImage()).cast_column('target_image', HFImage())
    
    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub"""
        if not self.config.push_to_hub:
            print("\n⊘ Skipping push to Hub (push_to_hub=False)")
            return
        
        print("\n" + "="*60)
        print("PUSHING TO HUB")
        print("="*60)
        
        try:
            print(f"\nRepository: {self.config.repo_id}")
            print(f"Split: {self.config.split}")
            print(f"Private: {self.config.private}")
            
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token
            )
            
            print(f"\n✓ Successfully pushed dataset to Hub!")
            print(f"  Dataset URL: https://huggingface.co/datasets/{self.config.repo_id}")
            
            # ✅ Upload original metadata files
            if self.config.upload_metadata:
                self._upload_metadata_to_hub()
            
        except Exception as e:
            print(f"\n✗ Error pushing to Hub: {e}")
            raise
    
    def _upload_metadata_to_hub(self) -> None:
        """
        Upload both results.json and results_checkpoint.json to Hub as dataset files
        Makes metadata accessible when exporting
        ✅ Handles both files independently with proper existence checks
        """
        # ✅ Get paths and check existence FIRST
        results_path = self._get_results_metadata_path("results.json")
        checkpoint_path = self._get_results_metadata_path("results_checkpoint.json")
        
        # If neither exists, skip
        if not results_path and not checkpoint_path:
            print("\n⚠ No metadata files (results.json or results_checkpoint.json) found")
            print("  Skipping metadata upload")
            return
        
        try:
            print("\n" + "="*60)
            print("UPLOADING ORIGINAL METADATA")
            print("="*60)
            
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # ✅ Upload results.json if it exists
            if results_path:
                try:
                    print(f"\nUploading results.json to {self.config.repo_id}...")
                    api.upload_file(
                        path_or_fileobj=str(results_path),
                        path_in_repo="results.json",
                        repo_id=self.config.repo_id,
                        repo_type="dataset",
                        token=self.config.token,
                        commit_message=f"Upload original results.json for split '{self.config.split}'"
                    )
                    print(f"  ✓ Successfully uploaded results.json!")
                    print(f"    File: https://huggingface.co/datasets/{self.config.repo_id}/blob/main/results.json")
                    
                except Exception as e:
                    print(f"  ✗ Error uploading results.json: {e}")
            
            # ✅ Upload results_checkpoint.json if it exists
            if checkpoint_path:
                try:
                    print(f"\nUploading results_checkpoint.json to {self.config.repo_id}...")
                    api.upload_file(
                        path_or_fileobj=str(checkpoint_path),
                        path_in_repo="results_checkpoint.json",
                        repo_id=self.config.repo_id,
                        repo_type="dataset",
                        token=self.config.token,
                        commit_message=f"Upload original results_checkpoint.json for split '{self.config.split}'"
                    )
                    print(f"  ✓ Successfully uploaded results_checkpoint.json!")
                    print(f"    File: https://huggingface.co/datasets/{self.config.repo_id}/blob/main/results_checkpoint.json")
                    
                except Exception as e:
                    print(f"  ✗ Error uploading results_checkpoint.json: {e}")
            
            # ✅ Log metadata statistics from results.json
            if results_path:
                try:
                    with open(results_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    num_generations = len(metadata.get('generations', []))
                    num_styles = len(metadata.get('styles', []))
                    num_chars = len(metadata.get('characters', []))
                    fonts = metadata.get('fonts', [])
                    
                    print(f"\n📊 Metadata Statistics:")
                    print(f"  Total generations: {num_generations}")
                    print(f"  Total styles: {num_styles}")
                    print(f"  Total characters: {num_chars}")
                    print(f"  Fonts: {', '.join(fonts) if fonts else 'unknown'}")
                except Exception as e:
                    print(f"⚠ Could not read metadata statistics: {e}")
            
        except ImportError:
            print("\n⚠ huggingface_hub not installed - skipping metadata upload")
            print("  Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"\n⚠ Warning: Could not upload metadata: {e}")
            print(f"  You can manually upload files to the Hub repository")
    
    def save_locally(self, output_path: str) -> None:
        """Save dataset and metadata locally for inspection"""
        print(f"\nSaving dataset locally to {output_path}")
        dataset = self.build_dataset()
        dataset.save_to_disk(output_path)
        print(f"✓ Dataset saved to {output_path}")
        
        # ✅ Also copy metadata files locally
        results_path = self._get_results_metadata_path("results.json")  # ✅ Fixed: added argument
        checkpoint_path = self._get_results_metadata_path("results_checkpoint.json")  # ✅ Fixed: added argument
        
        if results_path:
            import shutil
            local_results_path = Path(output_path) / "results.json"
            shutil.copy(results_path, local_results_path)
            print(f"✓ results.json saved to {local_results_path}")
        
        if checkpoint_path:
            import shutil
            local_checkpoint_path = Path(output_path) / "results_checkpoint.json"
            shutil.copy(checkpoint_path, local_checkpoint_path)
            print(f"✓ results_checkpoint.json saved to {local_checkpoint_path}")


def create_and_push_dataset(
    data_dir: str,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str] = None,
    upload_metadata: bool = True
) -> Dataset:
    """
    Create FontDiffusion dataset and optionally push to Hub
    
    Args:
        data_dir: Path to data_examples/train directory
        repo_id: Hugging Face repo ID (e.g., "username/fontdiffusion-dataset")
        split: Dataset split name (default: "train")
        push_to_hub: Whether to push to Hub
        private: Whether repo should be private
        token: HF token (if None, uses HUGGINGFACE_TOKEN env var)
        local_save_path: Path to save dataset locally
        upload_metadata: Whether to upload original results.json and results_checkpoint.json
    
    Returns:
        Dataset object
    """
    
    config = FontDiffusionDatasetConfig(
        data_dir=data_dir,
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        upload_metadata=upload_metadata
    )
    
    builder = FontDiffusionDatasetBuilder(config)
    
    # Build dataset
    dataset = builder.build_dataset()
    
    # Save locally if requested
    if local_save_path:
        builder.save_locally(local_save_path)
    
    # Push to Hub if requested (includes metadata upload)
    if push_to_hub:
        builder.push_to_hub(dataset)
    
    return dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and push FontDiffusion dataset")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data_examples/train directory')
    parser.add_argument('--repo_id', type=str, required=True,
                       help='Hugging Face repo ID (e.g., username/fontdiffusion-dataset)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split name')
    parser.add_argument('--private', action='store_true', default=False,
                       help='Make repository private')
    parser.add_argument('--no-push', action='store_true', default=False,
                       help='Do not push to Hub')
    parser.add_argument('--no-metadata', action='store_true', default=False,
                       help='Do not upload results.json metadata')
    parser.add_argument('--local-save', type=str, default=None,
                       help='Also save dataset locally to this path')
    parser.add_argument('--token', type=str, default=None,
                       help='Hugging Face token')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FONTDIFFUSION DATASET CREATOR")
    print("="*60)
    
    dataset = create_and_push_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        split=args.split,
        push_to_hub=not args.no_push,
        private=args.private,
        token=args.token,
        local_save_path=args.local_save,
        upload_metadata=not args.no_metadata  # ✅
    )
    
    print("\n✓ Done!")


"""
USAGE EXAMPLES:

# Upload with original results.json
python create_hf_dataset.py \
  --data_dir "my_dataset/train_original" \
  --repo_id "username/font-diffusion-generated-data" \
  --split "train_original" \
  --private \
  --token "hf_xxxxx"

# Upload without metadata
python create_hf_dataset.py \
  --data_dir "my_dataset/train" \
  --repo_id "username/font-diffusion-generated-data" \
  --split "train" \
  --no-metadata

# Save locally and upload
python create_hf_dataset.py \
  --data_dir "my_dataset/train" \
  --repo_id "username/font-diffusion-generated-data" \
  --split "train" \
  --local-save "exported_dataset/"

# Upload multiple splits
python create_hf_dataset.py --data_dir "my_dataset/train_original" --repo_id "username/font-diffusion-generated-data" --split "train_original" --token "hf_xxxxx"

python create_hf_dataset.py --data_dir "my_dataset/train" --repo_id "username/font-diffusion-generated-data" --split "train" --token "hf_xxxxx"

python create_hf_dataset.py --data_dir "my_dataset/val" --repo_id "username/font-diffusion-generated-data" --split "val" --token "hf_xxxxx"
"""