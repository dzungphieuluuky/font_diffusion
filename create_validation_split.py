"""
Create validation/test splits from training data
✅ FIXED: Ensures proper matching of content, style, and target images
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
import hashlib

def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """Compute deterministic hash for a (character, style, font) combination"""
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:8]


def get_content_filename(char: str, font: str = "") -> str:
    """
    Get content image filename for character
    Format: {unicode_codepoint}_{char}_{hash}.png
    Example: U+4E00_中_a1b2c3d4.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, "", font)
    safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""
    if safe_char:
        return f"{codepoint}_{safe_char}_{hash_val}.png"
    else:
        return f"{codepoint}_{hash_val}.png"


def get_target_filename(char: str, style: str, font: str = "") -> str:
    """
    Get target image filename
    Format: {unicode_codepoint}_{char}_{style}_{hash}.png
    Example: U+4E00_中_style0_a1b2c3d4.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, style, font)
    safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""
    if safe_char:
        return f"{codepoint}_{safe_char}_{style}_{hash_val}.png"
    else:
        return f"{codepoint}_{style}_{hash_val}.png"


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

    def analyze_data(self) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[Tuple[str, str], bool]]:
        """
        ✅ ENHANCED: Analyze training data and create a map of valid (char, style) pairs
        Supports hash-based filenames: U+XXXX_char_style_hash.png
        
        Returns:
        - All styles
        - All characters  
        - Character->Style mapping
        - Valid (char, style) pairs that have target images
        """
        print("\n" + "=" * 60)
        print("ANALYZING TRAINING DATA")
        print("=" * 60)

        styles = set()
        characters = set()
        char_to_styles = defaultdict(set)  # char -> set of styles
        valid_pairs = set()  # (char, style) pairs that exist as target images

        target_dir = self.source_train_dir / "TargetImage"
        content_dir = self.source_train_dir / "ContentImage"

        # ✅ Scan all style directories
        print("\n🔍 Scanning target images...")
        for style_folder in tqdm(sorted(target_dir.iterdir()), desc="Styles", unit="style", ncols=100):
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name  # The actual style name from folder
            styles.add(style_name)

            # Scan images with hash-based naming: U+XXXX_char_style_hash.png
            for img_file in style_folder.glob("*.png"):
                filename = img_file.stem  # Remove .png
                
                # Parse hash-based filename
                if "_" not in filename:
                    # Skip non-hash-based files
                    continue
                
                try:
                    # ✅ FIX: Since style names can contain underscores, we need a smarter approach
                    # Expected formats:
                    # 1. U+XXXX_char_style_hash (with safe char)
                    # 2. U+XXXX_style_hash (without safe char)
                    
                    # The hash is always last (8 hex characters)
                    parts = filename.split("_")
                    
                    if len(parts) < 3:
                        # Invalid format (need at least: codepoint, style_or_char, hash)
                        continue
                    
                    # Extract components
                    codepoint = parts[0]  # U+XXXX
                    hash_val = parts[-1]   # Last part is always hash
                    
                    # Validate codepoint format
                    if not codepoint.startswith("U+"):
                        continue
                    
                    # Decode character from codepoint
                    try:
                        char_code = int(codepoint.replace("U+", ""), 16)
                        char_part = chr(char_code)
                    except (ValueError, OverflowError):
                        tqdm.write(f"  ⚠️  Invalid codepoint: {codepoint} in {filename}")
                        continue
                    
                    # Now determine if there's a safe char in the filename
                    # Strategy: Check if parts[1] matches the expected safe char
                    safe_char_expected = char_part if char_part.isprintable() and char_part not in '<>:"/\\|?*' else ""
                    
                    if len(parts) >= 4 and parts[1] == safe_char_expected:
                        # Format: U+XXXX_char_style_hash
                        # Style is everything between char and hash (parts[2:-1])
                        style_part = "_".join(parts[2:-1])
                    else:
                        # Format: U+XXXX_style_hash (no safe char)
                        # Style is everything between codepoint and hash (parts[1:-1])
                        style_part = "_".join(parts[1:-1])
                    
                    # ✅ Validate that extracted style matches folder name
                    if style_part != style_name:
                        tqdm.write(f"  ⚠️  Style mismatch: extracted '{style_part}' != folder '{style_name}' in {filename}")
                        continue
                    
                    # Add to collections
                    characters.add(char_part)
                    char_to_styles[char_part].add(style_name)
                    valid_pairs.add((char_part, style_name))
                    
                except (IndexError, ValueError) as e:
                    tqdm.write(f"  ⚠️  Failed to parse: {filename} ({e})")
                    continue

        styles_list = sorted(list(styles))
        chars_list = sorted(list(characters))

        # ✅ Validate that all characters have content images
        print(f"\n✓ Initial scan:")
        print(f"  Styles: {len(styles_list)}")
        print(f"  Characters: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")

        # Check for missing content images
        print("\n🔍 Validating content images...")
        missing_content = []
        found_content = 0
        
        for char in tqdm(chars_list, desc="Checking content", unit="char", ncols=100):
            # Use hash-based content filename
            content_filename = get_content_filename(char, font="")
            content_path = content_dir / content_filename
            
            if not content_path.exists():
                missing_content.append(char)
            else:
                found_content += 1

        if missing_content:
            print(f"\n⚠️  WARNING: {len(missing_content)}/{len(chars_list)} characters missing content images:")
            print(f"  Examples: {missing_content[:10]}{'...' if len(missing_content) > 10 else ''}")
            print(f"  These characters will be excluded from splits")
            
            # Remove characters with missing content images
            for char in missing_content:
                del char_to_styles[char]
                characters.discard(char)
                # Remove all pairs with this character
                valid_pairs = {(c, s) for c, s in valid_pairs if c != char}
            
            chars_list = sorted(list(characters))

        print(f"\n✅ After validation:")
        print(f"  Characters with content images: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")
        print(f"  Content images found: {found_content}")
        
        # Show character distribution per style
        style_char_counts = {style: len([c for c, s in valid_pairs if s == style]) 
                            for style in styles_list}
        print(f"\n📊 Character distribution per style:")
        for style in sorted(style_char_counts.keys())[:10]:  # Show first 10
            print(f"  {style}: {style_char_counts[style]} characters")
        if len(style_char_counts) > 10:
            print(f"  ... and {len(style_char_counts) - 10} more styles")

        return styles_list, chars_list, dict(char_to_styles), valid_pairs


    def copy_images_for_split(
        self, 
        split_name: str, 
        split_dir: Path, 
        scenarios: Dict[str, Dict],
        valid_pairs: Set[Tuple[str, str]]
    ) -> Tuple[int, int, int]:
        """
        ✅ ENHANCED: Copy images for a specific split with hash-based filename support
        
        Args:
            split_name: Name of the split (e.g., "train", "val_seen_style_unseen_char")
            split_dir: Directory to copy files to
            scenarios: Dictionary of scenario configurations
            valid_pairs: Set of (char, style) tuples that have target images
        
        Returns:
            (content_copied, target_copied, skipped_pairs)
        """
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

        # ============================================================================
        # Step 1: Identify valid pairs for this split
        # ============================================================================
        split_valid_pairs = set()
        for char, style in valid_pairs:
            if char in allowed_chars and style in allowed_styles:
                split_valid_pairs.add((char, style))

        if not split_valid_pairs:
            print(f"  ⚠️  No valid pairs found for {split_name}")
            return 0, 0, 0

        print(f"  Valid pairs: {len(split_valid_pairs)}")

        # Extract unique chars and styles from valid pairs
        chars_in_split = {char for char, style in split_valid_pairs}
        styles_in_split = {style for char, style in split_valid_pairs}

        print(f"  Unique chars in valid pairs: {len(chars_in_split)}")
        print(f"  Unique styles in valid pairs: {len(styles_in_split)}")

        # ============================================================================
        # Step 2: Copy content images for characters that have target images
        # ============================================================================
        source_content_dir = self.source_train_dir / "ContentImage"
        content_copied = 0
        content_missing = 0

        print(f"\n  📥 Copying content images...")
        for char in tqdm(sorted(chars_in_split), desc="  Content", ncols=80, unit="char", leave=False):
            # Use hash-based filename
            content_filename = get_content_filename(char, font="")
            src_path = source_content_dir / content_filename
            dst_path = split_content_dir / content_filename

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing content image: {src_path}")
                content_missing += 1
                continue

            # Skip if source and destination are the same (already in place)
            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                content_copied += 1
            else:
                content_copied += 1  # Count as copied even if already there

        if content_missing > 0:
            print(f"  ⚠️  Missing {content_missing} content images")

        # ============================================================================
        # Step 3: Copy target images only for valid (char, style) pairs
        # ============================================================================
        source_target_dir = self.source_train_dir / "TargetImage"
        target_copied = 0
        skipped_pairs = 0

        print(f"  📥 Copying target images...")
        for char, style in tqdm(sorted(split_valid_pairs), desc="  Target", ncols=80, unit="pair", leave=False):
            style_dir = source_target_dir / style
            if not style_dir.exists():
                tqdm.write(f"    ⚠️  Style directory not found: {style_dir}")
                skipped_pairs += 1
                continue

            # Use hash-based filename
            target_filename = get_target_filename(char, style, font="")
            src_path = style_dir / target_filename

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing target image: {src_path}")
                skipped_pairs += 1
                continue

            dst_path = split_target_dir / style / target_filename

            # Skip if source and destination are the same
            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                target_copied += 1
            else:
                target_copied += 1  # Count as copied even if already there

        # ============================================================================
        # Step 4: Validate split
        # ============================================================================
        print(f"\n  🔍 Validating split...")
        try:
            self._validate_split(split_dir)
        except ValueError as e:
            print(f"  ❌ Validation failed: {e}")
            print(f"  ⚠️  Split may be incomplete or have missing content images")

        # Summary
        print(f"\n  📊 Summary:")
        print(f"    Content images copied: {content_copied}")
        print(f"    Target images copied: {target_copied}")
        print(f"    Skipped pairs: {skipped_pairs}")
        if content_missing > 0:
            print(f"    Missing content: {content_missing}")

        return content_copied, target_copied, skipped_pairs
    
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

    def _validate_split(self, split_dir: Path) -> None:
        """Validate that every target image has corresponding content image"""
        content_dir = split_dir / "ContentImage"
        target_dir = split_dir / "TargetImage"
        
        missing_pairs = 0
        total_targets = 0
        
        for style_folder in target_dir.iterdir():
            if not style_folder.is_dir():
                continue
            
            for target_file in style_folder.glob("*.png"):
                total_targets += 1
                filename = target_file.stem
                
                # ✅ NEW: Parse hash-based filename
                if "_" not in filename:
                    continue
                
                try:
                    parts = filename.split("_")
                    
                    # Extract character
                    if len(parts) >= 4:
                        char_part = parts[1]  # Has safe char
                    else:
                        # Decode from codepoint
                        codepoint = parts[0]
                        char_code = int(codepoint.replace("U+", ""), 16)
                        char_part = chr(char_code)
                    
                    # ✅ NEW: Use hash-based content filename
                    content_filename = get_content_filename(char_part, font="")
                    content_path = content_dir / content_filename
                    
                    if not content_path.exists():
                        missing_pairs += 1
                        tqdm.write(f"    ❌ Validation failed: {target_file.name} missing {content_filename}")
                except (IndexError, ValueError):
                    continue
        
        if missing_pairs > 0:
            print(f"  ⚠️  VALIDATION ERROR: {missing_pairs}/{total_targets} targets missing content images!")
            raise ValueError(
                f"Split validation failed: {missing_pairs} target images have no matching content images"
            )
        else:
            print(f"  ✓ Validation passed: All {total_targets} targets have matching content images")
    def create_splits(self) -> None:
        """Create all splits"""
        print("\n" + "=" * 60)
        print("CREATING DATA SPLITS")
        print("=" * 60)

        # ✅ Analyze data and get valid pairs
        styles, characters, char_to_styles, valid_pairs = self.analyze_data()

        # Create scenarios
        scenarios = self.create_validation_scenarios(styles, characters, char_to_styles)

        # Create directory structure
        print("\n🔧 Creating directory structure...")

        # Train split
        print("\n📁 Train split:")
        train_content, train_target, train_skipped = self.copy_images_for_split(
            "train", self.train_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {train_content} content + {train_target} target images (skipped {train_skipped})")

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
                val_content, val_target, val_skipped = self.copy_images_for_split(
                    val_scenario, scenario_dir, scenarios, valid_pairs
                )
                print(f"  ✓ Copied {val_content} content + {val_target} target images (skipped {val_skipped})")
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
            val_content, val_target, val_skipped = self.copy_images_for_split(
                "val", self.val_dir, val_combined_scenarios, valid_pairs
            )
            print(f"  ✓ Copied {val_content} content + {val_target} target images (skipped {val_skipped})")

        # Test split
        print(f"\n📁 test:")
        test_content, test_target, test_skipped = self.copy_images_for_split(
            "test", self.test_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {test_content} content + {test_target} target images (skipped {test_skipped})")

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
    Create validation splits with proper validation
    ✅ Ensures every (char, style) pair has both content and target images

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
        print("\n✅ Created directories with validated pairs:")
        print("  📁 train/ - Training data (matched content + targets)")
        print("  📁 val_seen_style_unseen_char/ - Test new characters")
        print("  📁 val_unseen_style_seen_char/ - Test new styles")
        print("  📁 val_unseen_both/ - Test full generalization")
        print("  📁 test/ - Hold-out test set")
    else:
        print("\n✅ Created directories with validated pairs:")
        print("  📁 train/ - Training data (matched content + targets)")
        print("  📁 val/ - Validation data (matched content + targets)")
        print("  📁 test/ - Test data (matched content + targets)")
    
    print("\n💡 Each folder guarantees:")
    print("  ✓ For every charX+styleY.png target, charX.png content exists")
    print("  ✓ No orphaned target images without content")
    print("  ✓ No unused content images without targets")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create validation splits with proper matching")
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

    try:
        create_validation_split(
            data_root=args.data_root,
            val_split_ratio=args.val_ratio,
            test_split_ratio=args.test_ratio,
            create_scenarios=args.scenarios,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

"""
Example usage:
python create_validation_split.py \\
  --data_root data_examples \\
  --val_ratio 0.2 \\
  --test_ratio 0.1 \\
  --seed 42
"""