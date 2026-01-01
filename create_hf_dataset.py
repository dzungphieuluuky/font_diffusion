"""
Create Hugging Face dataset from generated FontDiffusion images
✅ Uses hash-based file naming with unicode characters
✅ Relies on results_checkpoint.json as single source of truth
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Image as HFImage
from PIL import Image as PILImage
import pyarrow.parquet as pq
from huggingface_hub.utils import tqdm

from utilities import get_tqdm_config

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """
    Compute deterministic hash for a (character, style, font) combination

    Args:
        char: Unicode character
        style: Style name
        font: Font name (optional)

    Returns:
        8-character hash string
    """
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def get_content_filename(char: str, font: str = "") -> str:
    """
    Get content image filename for character
    Format: {unicode_codepoint}_{char}_{hash}.png or U+XXXX_{hash}.png
    Example: U+4E00_中_a1b2c3d4.png

    ✅ CORRECTED: Uses filesystem safety check, not isprintable()
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, "", font)

    # ✅ Filesystem-safe characters (remove problematic ones only)
    filesystem_unsafe = '<>:"/\\|?*'
    safe_char = char if char not in filesystem_unsafe else ""

    if safe_char:
        return f"{codepoint}_{safe_char}_{hash_val}.png"
    else:
        return f"{codepoint}_{hash_val}.png"


def get_target_filename(char: str, style: str, font: str = "") -> str:
    """
    Get target image filename
    Format: {unicode_codepoint}_{char}_{style}_{hash}.png or U+XXXX_{style}_{hash}.png
    Example: U+4E00_中_style0_a1b2c3d4.png

    ✅ CORRECTED: Uses filesystem safety check, not isprintable()
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, style, font)

    # ✅ Filesystem-safe characters (remove problematic ones only)
    filesystem_unsafe = '<>:"/\\|?*'
    safe_char = char if char not in filesystem_unsafe else ""

    if safe_char:
        return f"{codepoint}_{safe_char}_{style}_{hash_val}.png"
    else:
        return f"{codepoint}_{style}_{hash_val}.png"


# ============================================================================
# MAIN CLASS
# ============================================================================


@dataclass
class FontDiffusionDatasetConfig:
    """Configuration for dataset creation"""

    data_dir: str
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None


class FontDiffusionDatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format"""

    def __init__(self, config: FontDiffusionDatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.content_dir = self.data_dir / "ContentImage"
        self.target_dir = self.data_dir / "TargetImage"
        self.results_checkpoint = self.data_dir / "results_checkpoint.json"

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate directory structure"""
        if not self.content_dir.exists():
            raise ValueError(f"ContentImage directory not found: {self.content_dir}")
        if not self.target_dir.exists():
            raise ValueError(f"TargetImage directory not found: {self.target_dir}")
        if not self.results_checkpoint.exists():
            raise ValueError(
                f"results_checkpoint.json not found: {self.results_checkpoint}"
            )

        print(f"✓ Validated directory structure")
        print(f"  Content images: {self.content_dir}")
        print(f"  Target images: {self.target_dir}")
        print(f"  Results checkpoint: {self.results_checkpoint}")

    def _load_results_checkpoint(self) -> Dict[str, Any]:
        """Load results_checkpoint.json (single source of truth)"""
        with open(self.results_checkpoint, "r", encoding="utf-8") as f:
            results = json.load(f)

        print(f"\n✓ Loaded results_checkpoint.json")
        print(f"  Generations: {len(results.get('generations', []))}")
        print(f"  Characters: {len(results.get('characters', []))}")
        print(f"  Styles: {len(results.get('styles', []))}")

        return results

    def build_dataset(self) -> Dataset:
        """
        Build dataset from results_checkpoint.json
        ✅ Single source of truth
        """
        print("\n" + "=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)

        # Load checkpoint
        results = self._load_results_checkpoint()
        generations = results.get("generations", [])

        if not generations:
            raise ValueError("No generations found in results_checkpoint.json")

        dataset_rows: List[Dict[str, Any]] = []

        print(f"\n🖼️  Loading {len(generations)} image pairs...")

        # ✅ Use standardized tqdm config with total from loop
        for gen in tqdm(
            generations,
            **get_tqdm_config(
                total=len(generations),
                desc="Loading image pairs",
                unit="pair",
            ),
        ):
            char = gen.get("character")
            style = gen.get("style")
            font = gen.get("font", "unknown")

            # Get file paths from checkpoint
            content_path = self.data_dir / gen.get("content_image_path", "")
            target_path = self.data_dir / gen.get("target_image_path", "")

            # Verify files exist
            if not content_path.exists():
                tqdm.write(f"⚠️  Missing content: {content_path}")
                continue

            if not target_path.exists():
                tqdm.write(f"⚠️  Missing target: {target_path}")
                continue

            # Load images
            try:
                content_image = PILImage.open(content_path).convert("RGB")
                target_image = PILImage.open(target_path).convert("RGB")
            except Exception as e:
                tqdm.write(f"⚠️  Error loading pair ({char}, {style}): {e}")
                continue

            row = {
                "character": char,
                "style": style,
                "font": font,
                "content_image": content_image,
                "target_image": target_image,
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            }

            dataset_rows.append(row)

        print(f"✓ Loaded {len(dataset_rows)} samples")

        if not dataset_rows:
            raise ValueError("No samples loaded!")

        # Create HuggingFace dataset
        return (
            Dataset.from_dict(
                {
                    "character": [r["character"] for r in dataset_rows],
                    "style": [r["style"] for r in dataset_rows],
                    "font": [r["font"] for r in dataset_rows],
                    "content_image": [r["content_image"] for r in dataset_rows],
                    "target_image": [r["target_image"] for r in dataset_rows],
                    "content_hash": [r["content_hash"] for r in dataset_rows],
                    "target_hash": [r["target_hash"] for r in dataset_rows],
                }
            )
            .cast_column("content_image", HFImage())
            .cast_column("target_image", HFImage())
        )

    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub"""
        if not self.config.push_to_hub:
            print("\n⊘ Skipping push to Hub")
            return

        print("\n" + "=" * 60)
        print("PUSHING TO HUB")
        print("=" * 60)

        try:
            print(f"Repository: {self.config.repo_id}")
            print(f"Split: {self.config.split}")

            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token,
            )

            print(f"\n✓ Successfully pushed to Hub!")
            print(f"  URL: https://huggingface.co/datasets/{self.config.repo_id}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            raise

    def save_locally(self, output_path: str) -> None:
        """Save dataset locally"""
        print(f"\nSaving dataset to {output_path}")
        dataset = self.build_dataset()
        dataset.save_to_disk(output_path)
        print(f"✓ Saved!")


# ============================================================================
# ENTRY POINT
# ============================================================================


def create_and_push_dataset(
    data_dir: str,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str] = None,
) -> Dataset:
    """Create and optionally push dataset to Hub"""

    config = FontDiffusionDatasetConfig(
        data_dir=data_dir,
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
    )

    builder = FontDiffusionDatasetBuilder(config)
    dataset = builder.build_dataset()

    if local_save_path:
        builder.save_locally(local_save_path)

    if push_to_hub:
        builder.push_to_hub(dataset)

    return dataset


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Create HF dataset from FontDiffusion images"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to data directory (with ContentImage/ and TargetImage/)",
    )
    parser.add_argument(
        "--repo_id", type=str, required=True, help="HuggingFace repo ID"
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split name")
    parser.add_argument(
        "--private", action="store_true", default=False, help="Make repo private"
    )
    parser.add_argument(
        "--no-push", action="store_true", default=False, help="Don't push to Hub"
    )
    parser.add_argument(
        "--local-save", type=str, default=None, help="Also save locally to this path"
    )
    parser.add_argument("--token", type=str, default=None, help="HF token")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION DATASET CREATOR")
    print("=" * 60)
    print(f"\nData dir: {args.data_dir}")
    print(f"Repo: {args.repo_id}")
    print(f"Push to Hub: {not args.no_push}")

    try:
        create_and_push_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
        )

        print("\n✅ COMPLETE!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
