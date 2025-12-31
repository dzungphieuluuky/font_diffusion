"""
Create validation/test splits from training data
Supports multiple validation scenarios:
1. Seen char + Unseen style
2. Unseen char + Seen style
3. Unseen char + Unseen style
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import random

from tqdm import tqdm


@dataclass
class ValidationSplitConfig:
    """Configuration for validation split creation"""

    data_root: str  # e.g., "data_examples"
    val_split_ratio: float = 0.2  # 20% for validation
    test_split_ratio: float = 0.1  # 10% for test
    random_seed: int = 42
    create_scenarios: bool = True  # Create multiple validation scenarios


class ValidationSplitCreator:
    """Create train/val/test splits with different scenarios"""

    def __init__(self, config: ValidationSplitConfig):
        self.config = config
        self.data_root = Path(config.data_root)

        # Use separate directory for original data
        self.original_train_dir = self.data_root / "train_original"

        # Split directories
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"

        # Set random seed
        random.seed(config.random_seed)

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate training directory structure"""
        # Check if original exists, otherwise check train
        source_dir = (
            self.original_train_dir
            if self.original_train_dir.exists()
            else self.train_dir
        )

        if not (source_dir / "TargetImage").exists():
            raise ValueError(f"TargetImage not found in {source_dir}")
        if not (source_dir / "ContentImage").exists():
            raise ValueError(f"ContentImage not found in {source_dir}")

        self.source_train_dir = source_dir
        print(f"✓ Using source directory: {self.source_train_dir}")

    def analyze_data(self) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
        """
        Analyze training data to get:
        - All styles
        - All characters
        - Character->Style mapping
        """
        print("\n" + "=" * 60)
        print("ANALYZING TRAINING DATA")
        print("=" * 60)

        styles = set()
        characters = set()
        char_to_styles = defaultdict(set)  # char -> set of styles

        target_dir = self.source_train_dir / "TargetImage"

        # Scan all style directories
        for style_folder in target_dir.iterdir():
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name
            styles.add(style_name)

            # Scan images: style0+char0.png
            for img_file in style_folder.glob("*.png"):
                filename = img_file.stem  # Remove .png
                if "+" not in filename:
                    continue

                char_part = filename.split("+")[1]  # Get "char0"
                characters.add(char_part)
                char_to_styles[char_part].add(style_name)

        styles_list = sorted(list(styles))
        chars_list = sorted(list(characters))

        print(f"\nFound:")
        print(f"  Styles: {len(styles_list)} - {styles_list}")
        print(
            f"  Characters: {len(chars_list)} - {chars_list[:10]}{'...' if len(chars_list) > 10 else ''}"
        )

        return styles_list, chars_list, dict(char_to_styles)

    def create_validation_scenarios(
        self,
        styles: List[str],
        characters: List[str],
        char_to_styles: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        Create 4 validation scenarios:
        1. Seen char + Seen style (control - should be easy)
        2. Seen char + Unseen style (test generalization to new styles)
        3. Unseen char + Seen style (test generalization to new chars)
        4. Unseen char + Unseen style (test full generalization)
        """
        print("\n" + "=" * 60)
        print("CREATING VALIDATION SCENARIOS")
        print("=" * 60)

        num_styles = len(styles)
        num_chars = len(characters)

        # Split indices
        num_val_styles = max(1, int(num_styles * self.config.val_split_ratio))
        num_test_styles = max(1, int(num_styles * self.config.test_split_ratio))
        num_train_styles = num_styles - num_val_styles - num_test_styles

        num_val_chars = max(1, int(num_chars * self.config.val_split_ratio))
        num_test_chars = max(1, int(num_chars * self.config.test_split_ratio))
        num_train_chars = num_chars - num_val_chars - num_test_chars

        # Randomly split
        shuffled_styles = styles.copy()
        random.shuffle(shuffled_styles)

        shuffled_chars = characters.copy()
        random.shuffle(shuffled_chars)

        train_styles = set(shuffled_styles[:num_train_styles])
        val_styles = set(
            shuffled_styles[num_train_styles : num_train_styles + num_val_styles]
        )
        test_styles = set(shuffled_styles[num_train_styles + num_val_styles :])

        train_chars = set(shuffled_chars[:num_train_chars])
        val_chars = set(
            shuffled_chars[num_train_chars : num_train_chars + num_val_chars]
        )
        test_chars = set(shuffled_chars[num_train_chars + num_val_chars :])

        scenarios = {
            "train": {
                "styles": list(train_styles),
                "characters": list(train_chars),
                "description": "Seen styles + Seen characters (training data)",
            },
            "val_seen_style_unseen_char": {
                "styles": list(train_styles),
                "characters": list(val_chars),
                "description": "Seen styles + Unseen characters",
            },
            "val_unseen_style_seen_char": {
                "styles": list(val_styles),
                "characters": list(train_chars),
                "description": "Unseen styles + Seen characters",
            },
            "val_unseen_both": {
                "styles": list(val_styles),
                "characters": list(val_chars),
                "description": "Unseen styles + Unseen characters",
            },
            "test": {
                "styles": list(test_styles),
                "characters": list(test_chars),
                "description": "Test set (hold-out)",
            },
        }

        print("\n📊 Split Statistics:")
        print(
            f"  Styles: {num_train_styles} train + {num_val_styles} val + {num_test_styles} test"
        )
        print(
            f"  Chars:  {num_train_chars} train + {num_val_chars} val + {num_test_chars} test"
        )

        print("\n📋 Validation Scenarios:")
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n  {scenario_name}:")
            print(f"    Description: {scenario_data['description']}")
            print(f"    Styles: {scenario_data['styles']}")
            print(
                f"    Chars: {scenario_data['characters'][:5]}{'...' if len(scenario_data['characters']) > 5 else ''}"
            )

        return scenarios

    def copy_images_for_split(
        self, split_name: str, split_dir: Path, scenarios: Dict[str, Dict]
    ) -> int:
        """Copy images for a specific split"""
        split_config = scenarios[split_name]
        allowed_styles = set(split_config["styles"])
        allowed_chars = set(split_config["characters"])

        # Create directories
        split_content_dir = split_dir / "ContentImage"
        split_target_dir = split_dir / "TargetImage"
        split_content_dir.mkdir(parents=True, exist_ok=True)
        split_target_dir.mkdir(parents=True, exist_ok=True)

        # Create style subdirectories
        for style in allowed_styles:
            (split_target_dir / style).mkdir(exist_ok=True)

        # Copy content images
        source_content_dir = self.source_train_dir / "ContentImage"
        content_copied = 0

        for char in allowed_chars:
            src_path = source_content_dir / f"{char}.png"
            dst_path = split_content_dir / f"{char}.png"

            if src_path.exists() and src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                content_copied += 1

        # Copy target images
        source_target_dir = self.source_train_dir / "TargetImage"
        copied_count = 0

        for style in allowed_styles:
            style_dir = source_target_dir / style
            if not style_dir.exists():
                continue

            for img_file in style_dir.glob("*.png"):
                filename = img_file.stem
                if "+" not in filename:
                    continue

                char_part = filename.split("+")[1]
                if char_part in allowed_chars:
                    dst_path = split_target_dir / style / img_file.name

                    if img_file.resolve() != dst_path.resolve():
                        shutil.copy2(img_file, dst_path)
                        copied_count += 1

        print(f"  ✓ Copied {copied_count} target images")
        print(f"  ✓ Copied {content_copied} content images")

        return copied_count

    def create_splits(self) -> None:
        """Create all splits"""
        print("\n" + "=" * 60)
        print("CREATING DATA SPLITS")
        print("=" * 60)

        # Analyze data
        styles, characters, char_to_styles = self.analyze_data()

        # Create scenarios
        scenarios = self.create_validation_scenarios(styles, characters, char_to_styles)

        # Create directory structure
        print("\n🔧 Creating directory structure...")

        # Train split
        print("\n📁 Train split:")
        self.copy_images_for_split("train", self.train_dir, scenarios)

        # Validation splits
        if self.config.create_scenarios:
            val_scenarios = [
                "val_seen_style_unseen_char",
                "val_unseen_style_seen_char",
                "val_unseen_both",
            ]

            for val_scenario in val_scenarios:
                print(f"\n📁 {val_scenario}:")
                scenario_dir = self.data_root / val_scenario
                self.copy_images_for_split(val_scenario, scenario_dir, scenarios)
        else:
            # Create simple val directory (combination of all unseen)
            print(f"\n📁 val (all unseen):")
            val_combined_scenarios = {
                "val": {
                    "styles": scenarios["val_unseen_both"]["styles"]
                    + scenarios["val_unseen_style_seen_char"]["styles"],
                    "characters": scenarios["val_unseen_both"]["characters"]
                    + scenarios["val_unseen_style_seen_char"]["characters"],
                }
            }
            self.copy_images_for_split("val", self.val_dir, val_combined_scenarios)

        # Test split
        print(f"\n📁 test:")
        self.copy_images_for_split("test", self.test_dir, scenarios)

        # Save scenario metadata
        self._save_scenario_metadata(scenarios)

    def _save_scenario_metadata(self, scenarios: Dict[str, Dict]) -> None:
        """Save scenario information to JSON"""
        metadata_path = self.data_root / "validation_scenarios.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)

        print(f"\n✓ Saved scenario metadata to {metadata_path}")


def create_validation_split(
    data_root: str,
    val_split_ratio: float = 0.2,
    test_split_ratio: float = 0.1,
    create_scenarios: bool = True,
    random_seed: int = 42,
) -> None:
    """
    Create validation splits

    Args:
        data_root: Root data directory
        val_split_ratio: Fraction of data for validation
        test_split_ratio: Fraction of data for testing
        create_scenarios: Create separate scenario folders
        random_seed: Random seed for reproducibility
    """

    config = ValidationSplitConfig(
        data_root=data_root,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        random_seed=random_seed,
        create_scenarios=create_scenarios,
    )

    creator = ValidationSplitCreator(config)
    creator.create_splits()

    print("\n" + "=" * 60)
    print("✓ VALIDATION SPLIT CREATION COMPLETE")
    print("=" * 60)

    if create_scenarios:
        print("\nCreated directories:")
        print("  📁 train/ - Original training data")
        print("  📁 val_seen_style_unseen_char/ - Test new characters")
        print("  📁 val_unseen_style_seen_char/ - Test new styles")
        print("  📁 val_unseen_both/ - Test full generalization")
        print("  📁 test/ - Hold-out test set")
    else:
        print("\nCreated directories:")
        print("  📁 train/ - Training data")
        print("  📁 val/ - Validation data (combined unseen)")
        print("  📁 test/ - Test data")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create validation splits")
    parser.add_argument(
        "--data_root", type=str, default="data_examples", help="Root data directory"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        default=True,
        help="Create separate scenario folders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION VALIDATION SPLIT CREATOR")
    print("=" * 60)

    create_validation_split(
        data_root=args.data_root,
        val_split_ratio=args.val_ratio,
        test_split_ratio=args.test_ratio,
        create_scenarios=args.scenarios,
        random_seed=args.seed,
    )

"""Example
python create_validation_split.py \
  --data_root data_examples \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --seed 42
"""
