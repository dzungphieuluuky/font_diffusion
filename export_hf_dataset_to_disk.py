"""
Export Hugging Face dataset back to FontDiffusion directory structure.

This module reconstructs the original directory layout from a HuggingFace dataset,
preserving results_checkpoint.json as the single source of truth.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset
from PIL import Image
from tqdm.auto import tqdm

from filename_utils import compute_file_hash, get_content_filename, get_target_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export."""
    
    output_dir: Path
    repo_id: Optional[str] = None
    local_dataset_path: Optional[Path] = None
    split: str = "train"
    token: Optional[str] = None
    
    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.local_dataset_path, str):
            self.local_dataset_path = Path(self.local_dataset_path)
            
        if not self.repo_id and not self.local_dataset_path:
            raise ValueError(
                "Must provide either repo_id (Hub) or local_dataset_path (disk)"
            )


class DatasetExporter:
    """Export HuggingFace dataset to FontDiffusion directory structure."""
    
    def __init__(self, config: ExportConfig):
        """Initialize the exporter.
        
        Args:
            config: Export configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"
        
    def _load_dataset(self) -> Dataset:
        """Load dataset from Hub or local disk.
        
        Returns:
            Loaded dataset
            
        Raises:
            ValueError: If dataset cannot be loaded
        """
        if self.config.local_dataset_path:
            logger.info(f"Loading local dataset from {self.config.local_dataset_path}")
            try:
                dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
                logger.info(f"Loaded {len(dataset)} samples from disk")
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e
                
        logger.info(
            f"Loading dataset from Hub: {self.config.repo_id} (split: {self.config.split})"
        )
        try:
            dataset = load_dataset(
                self.config.repo_id,
                split=self.config.split,
                token=self.config.token,
            )
            logger.info(f"Loaded {len(dataset)} samples from Hub")
            return dataset
        except Exception as e:
            raise ValueError(
                f"Failed to load from Hub {self.config.repo_id}: {e}"
            ) from e
            
    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")
        
    def _export_images(self, dataset: Dataset) -> dict[str, Any]:
        """Export images and build metadata.
        
        Args:
            dataset: Dataset to export
            
        Returns:
            Complete metadata dictionary for results_checkpoint.json
        """
        logger.info("Exporting images...")
        
        exported_content = set()
        generations = []
        
        for sample in tqdm(dataset, desc="Exporting images", unit="sample"):
            char = sample["character"]
            style = sample["style"]
            font = sample.get("font", "unknown")
            
            # Export content image (once per character)
            content_filename = get_content_filename(char)
            if content_filename not in exported_content:
                content_img = sample.get("content_image")
                if isinstance(content_img, Image.Image):
                    content_path = self.content_dir / content_filename
                    content_img.save(content_path)
                    exported_content.add(content_filename)
                    
            # Export target image
            target_img = sample.get("target_image")
            if isinstance(target_img, Image.Image):
                style_dir = self.target_dir / style
                style_dir.mkdir(exist_ok=True)
                target_filename = get_target_filename(char, style)
                target_path = style_dir / target_filename
                target_img.save(target_path)
                
            # Build generation record
            generations.append({
                "character": char,
                "style": style,
                "font": font,
                "content_image_path": f"ContentImage/{content_filename}",
                "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            })
            
        logger.info(
            f"Exported {len(exported_content)} content images, "
            f"{len(generations)} target images"
        )
        
        # Build metadata
        characters = sorted({g["character"] for g in generations})
        styles = sorted({g["style"] for g in generations})
        fonts = sorted({g["font"] for g in generations if g["font"] != "unknown"})
        
        return {
            "generations": generations,
            "characters": characters,
            "styles": styles,
            "fonts": fonts if fonts else ["unknown"],
            "total_chars": len(characters),
            "total_styles": len(styles),
        }
        
    def _save_checkpoint(self, metadata: dict[str, Any]) -> None:
        """Save results_checkpoint.json.
        
        Args:
            metadata: Metadata dictionary to save
        """
        checkpoint_path = self.output_dir / "results_checkpoint.json"
        
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(
            f"Saved checkpoint with {len(metadata['generations'])} generations: "
            f"{len(metadata['characters'])} chars, {len(metadata['styles'])} styles"
        )
        
    def export(self) -> dict[str, Any]:
        """Execute the full export process.
        
        Returns:
            Complete metadata dictionary
            
        Raises:
            ValueError: If dataset loading or export fails
        """
        logger.info("Starting dataset export...")
        
        dataset = self._load_dataset()
        self._create_directories()
        metadata = self._export_images(dataset)
        self._save_checkpoint(metadata)
        
        logger.info("Export completed successfully")
        return metadata


def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    split: str = "train",
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Export HuggingFace dataset to disk.
    
    Args:
        output_dir: Directory to export to
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_dataset_path: Local dataset path (alternative to repo_id)
        split: Dataset split name (default: 'train')
        token: HuggingFace API token for private datasets
        
    Returns:
        Metadata dictionary from results_checkpoint.json
        
    Raises:
        ValueError: If neither repo_id nor local_dataset_path is provided,
                   or if dataset cannot be loaded
    """
    config = ExportConfig(
        output_dir=Path(output_dir),
        repo_id=repo_id,
        local_dataset_path=Path(local_dataset_path) if local_dataset_path else None,
        split=split,
        token=token,
    )
    
    exporter = DatasetExporter(config)
    return exporter.export()


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export HuggingFace dataset to FontDiffusion directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export from Hub
  python export_hf_dataset.py --output-dir ./output --repo-id user/dataset
  
  # Export from local cache
  python export_hf_dataset.py --output-dir ./output --local-path ~/.cache/...
        """,
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to export to",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local dataset path (alternative to --repo-id)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token for private datasets",
    )
    
    args = parser.parse_args()
    
    try:
        metadata = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            token=args.token,
        )
        
        logger.info(
            f"Successfully exported to {args.output_dir}\n"
            f"  ContentImage/\n"
            f"  TargetImage/\n"
            f"  results_checkpoint.json"
        )
        
    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()