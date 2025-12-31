"""
Create Hugging Face dataset from generated FontDiffusion images and push to Hub
✅ SIMPLIFIED: Only uses results_checkpoint.json (single source of truth)
"""

import os
import json
import shutil
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

    def _load_checkpoint_metadata(self) -> Optional[Dict[str, Any]]:
        """
        Load results_checkpoint.json metadata
        ✅ Single source of truth for all generation metadata

        Returns:
            Metadata dict if file exists and is valid, None otherwise
        """
        checkpoint_path = self.data_dir / "results_checkpoint.json"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"❌ results_checkpoint.json not found at {checkpoint_path}\n"
                f"   This file is required! Run sample_batch.py first to generate it."
            )

        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            num_gens = len(metadata.get("generations", []))
            print(f"✓ Loaded results_checkpoint.json ({num_gens} generations)")
            return metadata

        except json.JSONDecodeError as e:
            raise ValueError(
                f"❌ results_checkpoint.json is corrupted: {e}\n"
                f"   File: {checkpoint_path}"
            )
        except Exception as e:
            raise RuntimeError(f"❌ Error loading results_checkpoint.json: {e}")

    def build_dataset(self) -> Dataset:
        """
        Build dataset with structure matching sample_batch.py output:
        {
            'character': str,
            'char_index': int,
            'style': str,
            'style_index': int,
            'content_image': PIL.Image,
            'target_image': PIL.Image,
            'font': str
        }

        ✅ Loads images from paths in results_checkpoint.json
        """
        print("\n" + "=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)

        # ✅ Load ONLY results_checkpoint.json (required)
        metadata = self._load_checkpoint_metadata()

        dataset_rows: List[Dict[str, Any]] = []

        # Build mapping from generations
        gen_map: Dict[Tuple[int, int], Dict[str, Any]] = {}

        print(
            f"\n📋 Building index from {len(metadata['generations'])} generation records..."
        )

        for gen_info in metadata["generations"]:
            char_idx = gen_info.get("char_index")
            style_idx = gen_info.get("style_index")

            if char_idx is not None and style_idx is not None:
                gen_map[(char_idx, style_idx)] = gen_info

        # Iterate through style directories and load images
        print(f"\n🖼️  Loading images from disk...")

        for style_dir in sorted(self.target_dir.iterdir()):
            if not style_dir.is_dir():
                continue

            style_name = style_dir.name  # e.g., "style0"

            try:
                style_idx = int(style_name.replace("style", ""))
            except ValueError:
                continue

            for target_img_path in sorted(style_dir.glob("*.png")):
                # Parse filename: style0+char5.png
                filename = target_img_path.stem
                parts = filename.split("+")

                if len(parts) != 2:
                    continue

                char_idx_str = parts[1].replace("char", "")

                try:
                    char_idx = int(char_idx_str)
                except ValueError:
                    continue

                # Get content image path
                content_img_path = self.content_dir / f"char{char_idx}.png"

                if not content_img_path.exists():
                    tqdm.write(f"⚠ Missing content image: {content_img_path}")
                    continue

                if not target_img_path.exists():
                    tqdm.write(f"⚠ Missing target image: {target_img_path}")
                    continue

                # Load images from disk
                try:
                    content_image = PILImage.open(str(content_img_path)).convert("RGB")
                    target_image = PILImage.open(str(target_img_path)).convert("RGB")
                except Exception as e:
                    tqdm.write(f"⚠ Error loading images for {filename}: {e}")
                    continue

                # Get metadata for this pair from checkpoint
                gen_info = gen_map.get((char_idx, style_idx), {})

                # Extract information
                character = gen_info.get("character", f"char{char_idx}")
                font_name = gen_info.get("font", "unknown")

                row = {
                    "character": character,
                    "char_index": char_idx,
                    "style": style_name,
                    "style_index": style_idx,
                    "content_image": content_image,
                    "target_image": target_image,
                    "font": font_name,
                }

                dataset_rows.append(row)

        print(f"✓ Loaded {len(dataset_rows)} samples")

        if not dataset_rows:
            raise ValueError(
                "No samples loaded! Check that images exist in ContentImage/ and TargetImage/"
            )

        # Create HuggingFace dataset
        return (
            Dataset.from_dict(
                {
                    "character": [r["character"] for r in dataset_rows],
                    "char_index": [r["char_index"] for r in dataset_rows],
                    "style": [r["style"] for r in dataset_rows],
                    "style_index": [r["style_index"] for r in dataset_rows],
                    "content_image": [r["content_image"] for r in dataset_rows],
                    "target_image": [r["target_image"] for r in dataset_rows],
                    "font": [r["font"] for r in dataset_rows],
                }
            )
            .cast_column("content_image", HFImage())
            .cast_column("target_image", HFImage())
        )

    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub with metadata"""
        if not self.config.push_to_hub:
            print("\n⊘ Skipping push to Hub (push_to_hub=False)")
            return

        print("\n" + "=" * 60)
        print("PUSHING TO HUB")
        print("=" * 60)

        try:
            print(f"\nRepository: {self.config.repo_id}")
            print(f"Split: {self.config.split}")
            print(f"Private: {self.config.private}")

            # Push dataset
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token,
            )

            print(f"\n✓ Successfully pushed dataset to Hub!")
            print(
                f"  Dataset URL: https://huggingface.co/datasets/{self.config.repo_id}"
            )

            # ✅ Upload results_checkpoint.json
            self._upload_checkpoint_to_hub()

        except Exception as e:
            print(f"\n✗ Error pushing to Hub: {e}")
            raise

    def _upload_checkpoint_to_hub(self) -> None:
        """
        Upload results_checkpoint.json to Hub as dataset file
        ✅ Makes metadata accessible when exporting
        """
        checkpoint_path = self.data_dir / "results_checkpoint.json"

        if not checkpoint_path.exists():
            print("\n⚠ results_checkpoint.json not found - skipping upload")
            return

        try:
            print("\n" + "=" * 60)
            print("UPLOADING METADATA")
            print("=" * 60)

            from huggingface_hub import HfApi

            api = HfApi()

            print(f"\nUploading results_checkpoint.json to {self.config.repo_id}...")

            api.upload_file(
                path_or_fileobj=str(checkpoint_path),
                path_in_repo="results_checkpoint.json",
                repo_id=self.config.repo_id,
                repo_type="dataset",
                token=self.config.token,
                commit_message=f"Upload results_checkpoint.json for split '{self.config.split}'",
            )

            print(f"  ✓ Successfully uploaded results_checkpoint.json!")
            print(
                f"    File: https://huggingface.co/datasets/{self.config.repo_id}/blob/main/results_checkpoint.json"
            )

            # Log metadata stats
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self._log_metadata_stats(metadata)

        except ImportError:
            print("\n⚠ huggingface_hub not installed - skipping metadata upload")
            print("  Install with: pip install huggingface_hub")
        except Exception as e:
            print(f"\n⚠ Warning: Could not upload results_checkpoint.json: {e}")
            print(f"  You can manually upload the file to the Hub repository")

    def _log_metadata_stats(self, metadata: Dict[str, Any]) -> None:
        """Log metadata statistics"""
        try:
            num_generations = len(metadata.get("generations", []))
            num_styles = len(metadata.get("styles", []))
            num_chars = len(metadata.get("characters", []))
            fonts = metadata.get("fonts", [])

            print(f"\n📊 Metadata Statistics:")
            print(f"  Total generations: {num_generations}")
            print(f"  Total styles: {num_styles}")
            print(f"  Total characters: {num_chars}")
            print(f"  Fonts: {', '.join(fonts) if fonts else 'unknown'}")
        except Exception as e:
            print(f"⚠ Could not log metadata stats: {e}")

    def save_locally(self, output_path: str) -> None:
        """Save dataset and metadata locally for inspection"""
        print(f"\nSaving dataset locally to {output_path}")
        dataset = self.build_dataset()
        dataset.save_to_disk(output_path)
        print(f"✓ Dataset saved to {output_path}")

        # ✅ Copy results_checkpoint.json locally
        checkpoint_path = self.data_dir / "results_checkpoint.json"

        if checkpoint_path.exists():
            local_checkpoint_path = Path(output_path) / "results_checkpoint.json"
            shutil.copy(checkpoint_path, local_checkpoint_path)
            print(f"✓ results_checkpoint.json saved to {local_checkpoint_path}")
        else:
            print(f"⚠ results_checkpoint.json not found - skipping local copy")


def create_and_push_dataset(
    data_dir: str,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str] = None,
) -> Dataset:
    """
    Create FontDiffusion dataset and optionally push to Hub

    ✅ SIMPLIFIED: Only uses results_checkpoint.json

    Args:
        data_dir: Path to data_examples/train directory
        repo_id: Hugging Face repo ID (e.g., "username/fontdiffusion-dataset")
        split: Dataset split name (default: "train")
        push_to_hub: Whether to push to Hub
        private: Whether repo should be private
        token: HF token (if None, uses HUGGINGFACE_TOKEN env var)
        local_save_path: Path to save dataset locally

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
    import sys

    parser = argparse.ArgumentParser(
        description="Create and push FontDiffusion dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. Create and push to Hub:
   python create_hf_dataset.py \\
     --data_dir "my_dataset/train_original" \\
     --repo_id "username/font-diffusion-data" \\
     --split "train_original" \\
     --token "hf_xxxxx"

2. Create locally without pushing:
   python create_hf_dataset.py \\
     --data_dir "my_dataset/train" \\
     --repo_id "username/font-diffusion-data" \\
     --split "train" \\
     --no-push

3. Create and save locally:
   python create_hf_dataset.py \\
     --data_dir "my_dataset/train" \\
     --repo_id "username/font-diffusion-data" \\
     --split "train" \\
     --local-save "exported_dataset/"

4. Upload multiple splits to same repo:
   python create_hf_dataset.py \\
     --data_dir "my_dataset/train_original" \\
     --repo_id "username/font-diffusion-data" \\
     --split "train_original" \\
     --token "hf_xxxxx"

   python create_hf_dataset.py \\
     --data_dir "my_dataset/train" \\
     --repo_id "username/font-diffusion-data" \\
     --split "train" \\
     --token "hf_xxxxx"
        """,
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="✅ REQUIRED: Path to data_examples/train directory",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="✅ REQUIRED: Hugging Face repo ID (e.g., username/fontdiffusion-dataset)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split name (default: train)"
    )
    parser.add_argument(
        "--private", action="store_true", default=False, help="Make repository private"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        default=False,
        help="Do not push to Hub (only create locally)",
    )
    parser.add_argument(
        "--local-save",
        type=str,
        default=None,
        help="Also save dataset locally to this path",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (default: uses HUGGINGFACE_TOKEN env var)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION DATASET CREATOR")
    print("=" * 60)
    print(f"\n📊 Configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Repository: {args.repo_id}")
    print(f"  Split: {args.split}")
    print(f"  Private: {args.private}")
    print(f"  Push to Hub: {not args.no_push}")
    if args.local_save:
        print(f"  Local save: {args.local_save}")

    try:
        dataset = create_and_push_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
        )

        print("\n" + "=" * 60)
        print("✅ DATASET CREATION COMPLETE!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\n⚠️  Creation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
