"""
Export Hugging Face dataset back to original FontDiffusion directory structure
Preserves original results.json from pipeline generation
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import json
import os
import shutil

from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm


@dataclass
class ExportConfig:
    output_dir: str
    repo_id: Optional[str] = None
    local_dataset_path: Optional[str] = None
    split: str = "train"
    create_metadata: bool = True
    token: Optional[str] = None
    preserve_original_metadata: bool = True  # ✅ New flag


class DatasetExporter:
    """Export Hugging Face dataset to disk, preserving original metadata"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self) -> Dict[str, Any]:
        """
        Export dataset from Hub to disk
        
        Returns:
            Metadata dict (original or reconstructed)
        """
        print("\n" + "="*60)
        print("EXPORTING DATASET TO DISK")
        print("="*60)
        
        # Load dataset
        if self.config.local_dataset_path:
            print(f"Loading local dataset from {self.config.local_dataset_path}...")
            dataset = Dataset.load_from_disk(self.config.local_dataset_path)
        else:
            print(f"Loading dataset from Hub: {self.config.repo_id}#{self.config.split}...")
            dataset = load_dataset(
                self.config.repo_id,
                split=self.config.split,
                token=self.config.token
            )
        
        # Create directory structure
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        content_dir.mkdir(parents=True, exist_ok=True)
        target_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if original results.json exists in dataset
        original_metadata = self._try_load_original_metadata()
        
        if original_metadata and self.config.preserve_original_metadata:
            print("\n✅ Found original results.json metadata - preserving it")
            return self._export_with_original_metadata(dataset, original_metadata)
        else:
            print("\n⚠ Original metadata not found - reconstructing from dataset")
            return self._export_with_reconstructed_metadata(dataset)
    
    def _try_load_original_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Try to load original results.json from various sources
        1. From dataset card / extra files
        2. From Hub repository files
        """
        try:
            # Method 1: Check if results.json is in Hub repo files
            if self.config.repo_id:
                from huggingface_hub import hf_hub_download
                
                try:
                    metadata_path = hf_hub_download(
                        repo_id=self.config.repo_id,
                        filename="results.json",
                        repo_type="dataset",
                        token=self.config.token
                    )
                    
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    print(f"✓ Loaded original results.json from Hub")
                    return metadata
                except Exception as e:
                    print(f"  (results.json not in Hub: {e})")
            
            # Method 2: Check local cache
            local_cache_path = self.output_dir.parent / "results.json"
            if local_cache_path.exists():
                with open(local_cache_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                print(f"✓ Loaded original results.json from local cache")
                return metadata
            
            return None
        except Exception as e:
            print(f"  Could not load original metadata: {e}")
            return None
    
    def _export_with_original_metadata(self, 
                                       dataset: Dataset, 
                                       original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export dataset while preserving original results.json structure
        Ensures output paths match actual files
        """
        print("\n" + "="*60)
        print("EXPORTING WITH ORIGINAL METADATA")
        print("="*60)
        
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        # Export content images
        print("\nExporting content images...")
        content_samples_exported = 0
        
        for idx, sample in enumerate(tqdm(dataset, desc="📝 Exporting content images", ncols=80)):
            if 'content_image' in sample:
                char_idx = sample.get('char_index', idx)
                content_img = sample['content_image']
                
                if isinstance(content_img, PILImage.Image):
                    content_path = content_dir / f"char{char_idx}.png"
                    content_img.save(str(content_path))
                    content_samples_exported += 1
        
        print(f"✓ Exported {content_samples_exported} content images")
        
        # Export target images organized by style
        print("\nExporting target images...")
        target_samples_exported = 0
        
        # Create style directories and track exports
        for sample in tqdm(dataset, desc="🎨 Exporting target images", ncols=80):
            if 'target_image' in sample:
                char_idx = sample.get('char_index', 0)
                style = sample.get('style', 'style0')
                
                style_dir = target_base_dir / style
                style_dir.mkdir(parents=True, exist_ok=True)
                
                target_img = sample['target_image']
                
                if isinstance(target_img, PILImage.Image):
                    target_path = style_dir / f"{style}+char{char_idx}.png"
                    target_img.save(str(target_path))
                    target_samples_exported += 1
        
        print(f"✓ Exported {target_samples_exported} target images")
        
        # ✅ Update original metadata with correct file paths
        updated_metadata = self._update_metadata_paths(original_metadata)
        
        # Save updated metadata
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(updated_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved original results.json to {results_path}")
        
        return updated_metadata
    
    def _update_metadata_paths(self, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update file paths in original metadata to match export directory
        """
        metadata = json.loads(json.dumps(original_metadata))  # Deep copy
        
        # Normalize paths
        for generation in metadata.get('generations', []):
            # Update content image path
            if 'content_image_path' in generation:
                char_idx = generation.get('char_index', 0)
                generation['content_image_path'] = f"{self.config.output_dir}/ContentImage/char{char_idx}.png"
            
            # Update target image path
            if 'target_image_path' in generation:
                char_idx = generation.get('char_index', 0)
                style = generation.get('style', 'style0')
                generation['target_image_path'] = f"{self.config.output_dir}/TargetImage/{style}/{style}+char{char_idx}.png"
        
        return metadata
    
    def _export_with_reconstructed_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Reconstruct metadata from dataset if original not available
        ⚠ Use only as fallback
        """
        print("\n" + "="*60)
        print("RECONSTRUCTING METADATA FROM DATASET")
        print("="*60)
        
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        # Export images
        print("\nExporting images...")
        content_samples_exported = 0
        target_samples_exported = 0
        
        metadata: Dict[str, Any] = {
            'generations': [],
            'metrics': {},
            'styles': [],
            'characters': [],
            'total_chars': 0,
            'total_styles': 0
        }
        
        unique_chars = set()
        unique_styles = set()
        
        for sample in tqdm(dataset, desc="📊 Exporting samples", ncols=80):
            char_idx = sample.get('char_index', 0)
            character = sample.get('character', '?')
            style = sample.get('style', 'style0')
            style_idx = sample.get('style_index', 0)
            font = sample.get('font', 'unknown')
            
            unique_chars.add(character)
            unique_styles.add(style)
            
            # Export content image
            if 'content_image' in sample:
                content_img = sample['content_image']
                if isinstance(content_img, PILImage.Image):
                    content_path = content_dir / f"char{char_idx}.png"
                    content_img.save(str(content_path))
                    content_samples_exported += 1
            
            # Export target image
            if 'target_image' in sample:
                target_img = sample['target_image']
                if isinstance(target_img, PILImage.Image):
                    style_dir = target_base_dir / style
                    style_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_path = style_dir / f"{style}+char{char_idx}.png"
                    target_img.save(str(target_path))
                    target_samples_exported += 1
            
            # Add to metadata
            metadata['generations'].append({
                'character': character,
                'char_index': char_idx,
                'style': style,
                'style_index': style_idx,
                'font': font,
                'content_image_path': f"{self.config.output_dir}/ContentImage/char{char_idx}.png",
                'target_image_path': f"{self.config.output_dir}/TargetImage/{style}/{style}+char{char_idx}.png"
            })
        
        # Update metadata summary
        metadata['characters'] = sorted(list(unique_chars))
        metadata['styles'] = sorted(list(unique_styles))
        metadata['total_chars'] = len(unique_chars)
        metadata['total_styles'] = len(unique_styles)
        
        print(f"✓ Exported {content_samples_exported} content images")
        print(f"✓ Exported {target_samples_exported} target images")
        
        # Save metadata
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Saved reconstructed results.json to {results_path}")
        
        return metadata


def export_dataset(
    output_dir: str,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    create_metadata: bool = True,
    token: Optional[str] = None,
    preserve_original: bool = True
) -> None:
    """
    Export dataset from Hub or local path to disk
    
    Args:
        output_dir: Directory to export to
        repo_id: Hub repo ID
        local_dataset_path: Local dataset path
        split: Dataset split
        create_metadata: Whether to create metadata
        token: HF token
        preserve_original: Try to preserve original results.json
    """
    
    config = ExportConfig(
        output_dir=output_dir,
        repo_id=repo_id,
        local_dataset_path=local_dataset_path,
        split=split,
        create_metadata=create_metadata,
        token=token,
        preserve_original_metadata=preserve_original
    )
    
    exporter = DatasetExporter(config)
    metadata = exporter.export()
    
    print("\n" + "="*60)
    print("✓ EXPORT COMPLETE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── ContentImage/")
    print(f"    │   ├── char0.png")
    print(f"    │   ├── char1.png")
    print(f"    │   └── ...")
    print(f"    ├── TargetImage/")
    print(f"    │   ├── style0/")
    print(f"    │   ├── style1/")
    print(f"    │   └── ...")
    print(f"    └── results.json (original or reconstructed)")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Hugging Face dataset to disk")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to export to')
    parser.add_argument('--repo_id', type=str, default=None,
                       help='Hugging Face repo ID')
    parser.add_argument('--local_dataset_path', type=str, default=None,
                       help='Local dataset path')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split')
    parser.add_argument('--token', type=str, default=None,
                       help='Hugging Face token')
    parser.add_argument('--preserve_original', action='store_true', default=True,
                       help='Preserve original results.json from pipeline')
    parser.add_argument('--no_preserve_original', action='store_false', dest='preserve_original',
                       help='Do not preserve original results.json')
    
    args = parser.parse_args()
    
    export_dataset(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        local_dataset_path=args.local_dataset_path,
        split=args.split,
        token=args.token,
        preserve_original=args.preserve_original
    )

"""Example

# Export with original metadata preserved
python export_hf_dataset_to_disk.py \
  --output_dir "my_dataset/train_original" \
  --repo_id "dzungpham/font-diffusion-generated-data" \
  --split "train_original" \
  --token "hf_xxxxx" \
  --preserve_original

# Or without preservation (fallback to reconstruction)
python export_hf_dataset_to_disk.py \
  --output_dir "my_dataset/train" \
  --repo_id "dzungpham/font-diffusion-generated-data" \
  --split "train" \
  --no_preserve_original

  """