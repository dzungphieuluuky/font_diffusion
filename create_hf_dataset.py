"""
Create Hugging Face dataset from generated FontDiffusion images.

This module builds datasets from FontDiffusion outputs, using results_checkpoint.json
as the single source of truth for generation metadata.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image
from tqdm.auto import tqdm
from utilities import get_hf_bar

from filename_utils import compute_file_hash

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    data_dir: Path
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None

    def __post_init__(self):
        """Convert data_dir to Path if it's a string."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


class DatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format."""

    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset builder.

        Args:
            config: Dataset configuration

        Raises:
            ValueError: If directory structure is invalid
        """
        self.config = config
        self.data_dir = config.data_dir
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that all required directories and files exist.

        Raises:
            ValueError: If any required directory or file is missing
        """
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")

        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        logger.info("Directory structure validated successfully")

    def _load_checkpoint(self) -> dict[str, Any]:
        """Load and validate results checkpoint.

        Returns:
            Checkpoint data dictionary

        Raises:
            ValueError: If checkpoint is invalid or empty
        """
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE

        with checkpoint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")

        logger.info(
            f"Loaded checkpoint: {len(generations)} generations, "
            f"{len(data.get('characters', []))} characters, "
            f"{len(data.get('styles', []))} styles"
        )

        return data

    def build(self) -> Dataset:
        """Build the dataset from checkpoint data.

        Returns:
            HuggingFace Dataset with image pairs and metadata

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info("Building dataset...")

        checkpoint = self._load_checkpoint()
        generations = checkpoint["generations"]

        # Pre-allocate lists for better performance
        characters = []
        styles = []
        fonts = []
        content_images = []
        target_images = []
        content_hashes = []
        target_hashes = []

        skipped = 0

        for gen in get_hf_bar(
            generations, 
            desc="Loading image pairs", 
            unit="pair"
        ):
            char = gen.get("character")
            style = gen.get("style")
            font = gen.get("font", "unknown")

            # Construct paths
            content_path = self.data_dir / gen.get("content_image_path", "")
            target_path = self.data_dir / gen.get("target_image_path", "")

            # Validate paths exist
            if not content_path.exists() or not target_path.exists():
                skipped += 1
                continue

            # Load images
            try:
                content_img = Image.open(content_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load images for {char}/{style}: {e}")
                skipped += 1
                continue

            # Append to lists
            characters.append(char)
            styles.append(style)
            fonts.append(font)
            content_images.append(content_img)
            target_images.append(target_img)
            content_hashes.append(compute_file_hash(char, "", font))
            target_hashes.append(compute_file_hash(char, style, font))

        if not characters:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully loaded {len(characters)} samples")

        # Define explicit features for better type safety
        features = Features(
            {
                "character": Value("string"),
                "style": Value("string"),
                "font": Value("string"),
                "content_image": HFImage(),
                "target_image": HFImage(),
                "content_hash": Value("string"),
                "target_hash": Value("string"),
            }
        )

        # Create dataset
        dataset = Dataset.from_dict(
            {
                "character": characters,
                "style": styles,
                "font": fonts,
                "content_image": content_images,
                "target_image": target_images,
                "content_hash": content_hashes,
                "target_hash": target_hashes,
            },
            features=features,
        )

        return dataset

    def push(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub.

        Args:
            dataset: Dataset to push
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Pushing dataset to {self.config.repo_id}...")

        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            split=self.config.split,
            private=self.config.private,
            token=self.config.token,
        )

        logger.info(
            f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id}"
        )

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk.

        Args:
            dataset: Dataset to save
            output_path: Local directory path
        """
        logger.info(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved successfully")


def create_dataset(
    data_dir: str | Path,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
) -> Dataset:
    """Create and optionally push dataset to Hub.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name (default: 'train')
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        private: Whether to make the repository private (default: False)
        token: HuggingFace API token (optional)
        local_save_path: Local path to save dataset (optional)

    Returns:
        Created Dataset object

    Raises:
        ValueError: If data directory structure is invalid or no samples found
    """
    config = DatasetConfig(
        data_dir=Path(data_dir),
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
    )

    builder = DatasetBuilder(config)
    dataset = builder.build()

    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push(dataset)

    return dataset


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from FontDiffusion images"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (with ContentImage/ and TargetImage/)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (username/dataset-name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to Hub",
    )
    parser.add_argument(
        "--local-save",
        type=str,
        help="Save dataset locally to this path",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token",
    )

    args = parser.parse_args()

    try:
        create_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
