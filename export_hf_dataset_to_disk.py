"""
Export Hugging Face dataset back to original FontDiffusion directory structure
✅ Uses hash-based file naming with unicode characters
✅ Preserves results_checkpoint.json as single source of truth
"""

from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import hashlib

from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from huggingface_hub.utils import tqdm
import json
import os

from filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash
    )

@dataclass
class ExportConfig:
    """Configuration for dataset export"""

    output_dir: str
    repo_id: Optional[str] = None
    local_dataset_path: Optional[str] = None
    split: str = "train"
    token: Optional[str] = None


class DatasetExporter:
    """Export Hugging Face dataset to disk"""

    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)

    def export(self) -> Dict[str, Any]:
        """
        Export dataset from Hub to disk

        Returns:
            metadata: The complete results_checkpoint.json data
        """
        print("\n" + "=" * 60)
        print("EXPORTING DATASET TO DISK")
        print("=" * 60)

        # Load dataset
        if self.config.local_dataset_path:
            print(f"Loading local dataset from {self.config.local_dataset_path}...")
            try:
                dataset = Dataset.load_from_disk(self.config.local_dataset_path)
                print(f"✓ Loaded dataset with {len(dataset)} samples")
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}")

        elif self.config.repo_id:
            print(f"\n📥 Loading dataset from Hub...")
            print(f"   Repository: {self.config.repo_id}")
            print(f"   Split: {self.config.split}")

            try:
                dataset = load_dataset(
                    self.config.repo_id,
                    split=self.config.split,
                    token=self.config.token,
                )
                print(f"✓ Loaded dataset with {len(dataset)} samples from Hub")

            except Exception as e:
                print(f"\n❌ Error loading from Hub:")
                print(f"   Repository: {self.config.repo_id}")
                print(f"   Split: {self.config.split}")
                print(f"   Error: {type(e).__name__}: {e}")
                raise

        else:
            raise ValueError(
                "❌ Must provide either:\n"
                "   --repo_id (load from Hub)\n"
                "   --local_dataset_path (load from disk)"
            )

        # Create directory structure
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"

        content_dir.mkdir(parents=True, exist_ok=True)
        target_base_dir.mkdir(parents=True, exist_ok=True)

        # Export images and build metadata
        return self._export_images_and_build_metadata(dataset)

    def _export_images_and_build_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        """Export images and build metadata"""

        print("\nExporting images from dataset...")

        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"

        # Track exported content images to avoid duplicates
        exported_content = set()

        # Build generations list
        generations = []

        # Export images
        print("\n🎨 Exporting images...")
        for sample in tqdm(dataset, desc="Exporting", ncols=80):
            char = sample.get("character")
            style = sample.get("style")
            font = sample.get("font", "unknown")

            # Export content image (once per character)
            content_filename = get_content_filename(char)

            if content_filename not in exported_content:
                if "content_image" in sample:
                    content_img = sample["content_image"]
                    if isinstance(content_img, PILImage.Image):
                        content_path = content_dir / content_filename
                        content_img.save(str(content_path))
                        exported_content.add(content_filename)

            # Export target image
            if "target_image" in sample:
                style_dir = target_base_dir / style
                style_dir.mkdir(parents=True, exist_ok=True)

                target_filename = get_target_filename(char, style)
                target_img = sample["target_image"]

                if isinstance(target_img, PILImage.Image):
                    target_path = style_dir / target_filename
                    target_img.save(str(target_path))

            # Build generation record
            generations.append(
                {
                    "character": char,
                    "style": style,
                    "font": font,
                    "content_image_path": f"ContentImage/{get_content_filename(char)}",
                    "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
                    "content_hash": compute_file_hash(char, "", font),
                    "target_hash": compute_file_hash(char, style, font),
                }
            )

        print(f"✓ Exported {len(exported_content)} content images")
        print(f"✓ Exported {len(generations)} target images")

        # Build metadata
        characters_set = set(g["character"] for g in generations)
        styles_set = set(g["style"] for g in generations)
        fonts_set = set(g["font"] for g in generations if g["font"] != "unknown")

        metadata = {
            "generations": generations,
            "characters": sorted(list(characters_set)),
            "styles": sorted(list(styles_set)),
            "fonts": sorted(list(fonts_set)) if fonts_set else ["unknown"],
            "total_chars": len(characters_set),
            "total_styles": len(styles_set),
        }

        # Save results_checkpoint.json
        print("\n💾 Saving results_checkpoint.json...")
        checkpoint_path = self.output_dir / "results_checkpoint.json"
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved results_checkpoint.json ({len(generations)} generations)")
        self._log_metadata_stats(metadata)

        return metadata

    def _log_metadata_stats(self, metadata: Dict[str, Any]) -> None:
        """Log metadata statistics"""
        print("\n📊 Metadata Statistics:")
        print(f"  Total generations: {len(metadata.get('generations', []))}")
        print(f"  Total characters: {metadata.get('total_chars', 0)}")
        print(f"  Total styles: {metadata.get('total_styles', 0)}")
        print(f"  Fonts: {', '.join(metadata.get('fonts', ['unknown']))}")


def export_dataset(
    output_dir: str,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    token: Optional[str] = None,
) -> None:
    """Export dataset to disk"""

    config = ExportConfig(
        output_dir=output_dir,
        repo_id=repo_id,
        local_dataset_path=local_dataset_path,
        split=split,
        token=token,
    )

    exporter = DatasetExporter(config)
    metadata = exporter.export()

    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles created:")
    print(f"  ✓ {output_dir}/ContentImage/")
    print(f"  ✓ {output_dir}/TargetImage/")
    print(f"  ✓ {output_dir}/results_checkpoint.json")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export Hugging Face dataset to disk",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:

1. Export from Hub:
   python export_hf_dataset_to_disk.py \\
     --output_dir "my_dataset/train" \\
     --repo_id "dzungpham/font-diffusion-data" \\
     --split "train"

2. Export from local cache:
   python export_hf_dataset_to_disk.py \\
     --output_dir "my_dataset/train" \\
     --local_dataset_path "~/.cache/huggingface/datasets/.../train"
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="✅ REQUIRED: Directory to export to",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Hugging Face repo ID (e.g., username/dataset-name)",
    )
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default=None,
        help="Local dataset path (alternative to --repo_id)",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split name (default: train)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (for private datasets)",
    )

    args = parser.parse_args()

    # Validation
    if not args.repo_id and not args.local_dataset_path:
        print("\n" + "=" * 60)
        print("❌ ERROR: Missing required argument")
        print("=" * 60)
        print("\nYou must provide EITHER:")
        print("  --repo_id          : Load from Hugging Face Hub")
        print("  --local_dataset_path : Load from local disk")
        print("=" * 60)
        sys.exit(1)

    try:
        export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_dataset_path,
            split=args.split,
            token=args.token,
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Export failed:")
        print(f"   {type(e).__name__}: {e}")
        sys.exit(1)
