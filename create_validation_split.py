"""
Create validation/test splits from training data
✅ SIMPLIFIED: Creates only train + val (unseen char + unseen style)
✅ Parses filenames by codepoint and hash only (no reliance on safe char)
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

from tqdm import tqdm
import hashlib


def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """Compute deterministic hash for a (character, style, font) combination"""
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]


def get_content_filename(char: str, font: str = "") -> str:
    """
    Get content image filename for character
    Format: {unicode_codepoint}_{char}_{hash}.png or {unicode_codepoint}_{hash}.png
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
    Format: {unicode_codepoint}_{char}_{style}_{hash}.png or {unicode_codepoint}_{style}_{hash}.png
    """
    codepoint = f"U+{ord(char):04X}"
    hash_val = compute_file_hash(char, style, font)
    safe_char = char if char.isprintable() and char not in '<>:"/\\|?*' else ""
    if safe_char:
        return f"{codepoint}_{safe_char}_{style}_{hash_val}.png"
    else:
        return f"{codepoint}_{style}_{hash_val}.png"


def detect_font_parameter(content_dir: Path, sample_chars: List[str]) -> str:
    """
    Auto-detect the font parameter used during generation by reverse-engineering
    from existing content image filenames.
    """
    print("\n🔍 Auto-detecting font parameter from existing files...")
    
    # ✅ Get fonts from fonts/ directory
    fonts_dir = Path(__file__).parent / "fonts"
    candidate_fonts = ["", "default"]  # Start with empty and default
    
    if fonts_dir.exists() and fonts_dir.is_dir():
        # Add all .ttf and .otf files (without extension)
        for font_file in fonts_dir.glob("*.[to][tf][f]"):  # Matches .ttf and .otf
            font_name = font_file.stem  # Filename without extension
            candidate_fonts.append(font_name)
        
        print(f"  📂 Found {len(candidate_fonts) - 2} fonts in fonts/ directory")
    else:
        print(f"  ⚠️  fonts/ directory not found, using defaults only")
    
    # Try to find a match using sample characters
    for test_char in sample_chars[:5]:  # Test with first 5 chars
        # Get actual filename from disk using only codepoint
        codepoint = f"U+{ord(test_char):04X}"
        
        # Find files matching this character's codepoint (pattern: U+XXXX_*.png)
        pattern = f"{codepoint}_*.png"
        matching_files = list(content_dir.glob(pattern))
        
        if not matching_files:
            continue
        
        # Get the actual hash from the filename
        actual_file = matching_files[0]
        actual_filename = actual_file.stem
        parts = actual_filename.split("_")
        
        if len(parts) >= 2:
            actual_hash = parts[-1]  # Last part is the hash
            
            # Try each candidate font to see which produces this hash
            for candidate_font in candidate_fonts:
                test_hash = compute_file_hash(test_char, "", candidate_font)
                
                if test_hash == actual_hash:
                    print(f"  ✓ Detected font parameter: '{candidate_font}'")
                    print(f"    Verified with: {actual_file.name}")
                    return candidate_font
    
    # If no match found, show some example filenames for manual inspection
    print(f"  ⚠️  Could not auto-detect font parameter")
    print(f"  Example files found in {content_dir}:")
    
    example_files = list(content_dir.glob("*.png"))[:5]
    for f in example_files:
        print(f"    {f.name}")
    
    print(f"\n  Available candidate fonts tried:")
    for cf in candidate_fonts[:10]:
        print(f"    '{cf}'")
    if len(candidate_fonts) > 10:
        print(f"    ... and {len(candidate_fonts) - 10} more")
    
    print(f"\n  Please check the hash generation in your sample script.")
    print(f"  Defaulting to empty string ('')")
    
    return ""

@dataclass
class ValidationSplitConfig:
    """Configuration for validation split creation"""

    data_root: str  # e.g., "data_examples"
    val_split_ratio: float = 0.2  # 20% for validation
    random_seed: int = 42


class ValidationSplitCreator:
    """Create train/val splits (unseen char + unseen style only)"""

    def __init__(self, config: ValidationSplitConfig):
        self.config = config
        self.data_root = Path(config.data_root)

        # Use separate directory for original data
        self.original_train_dir = self.data_root / "train_original"

        # Split directories
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"

        # Set random seed
        random.seed(config.random_seed)
        
        # Will be detected from files
        self.detected_font = None

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

    def analyze_data(
        self,
    ) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[Tuple[str, str], bool]]:
        """
        ✅ SIMPLIFIED: Analyze training data by parsing codepoint and hash only
        Does not rely on safe character in filename

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
        char_to_styles = defaultdict(set)
        valid_pairs = set()

        target_dir = self.source_train_dir / "TargetImage"
        content_dir = self.source_train_dir / "ContentImage"

        # ✅ Scan all style directories
        print("\n🔍 Scanning target images...")
        for style_folder in tqdm(
            sorted(target_dir.iterdir()), desc="Styles", unit="style", ncols=100
        ):
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name  # The actual style name from folder
            styles.add(style_name)

            # Scan images: U+XXXX_[char]_style_hash.png or U+XXXX_style_hash.png
            for img_file in style_folder.glob("*.png"):
                filename = img_file.stem  # Remove .png

                if "_" not in filename:
                    continue

                try:
                    parts = filename.split("_")
                    
                    if len(parts) < 3:
                        continue

                    # ✅ SIMPLIFIED: Extract only codepoint and hash
                    codepoint = parts[0]  # U+XXXX
                    hash_val = parts[-1]  # Last part is always hash

                    # Validate codepoint format
                    if not codepoint.startswith("U+"):
                        continue

                    # ✅ Decode character from codepoint (no reliance on safe char)
                    try:
                        char_code = int(codepoint.replace("U+", ""), 16)
                        char_part = chr(char_code)
                    except (ValueError, OverflowError):
                        tqdm.write(f"  ⚠️  Invalid codepoint: {codepoint} in {filename}")
                        continue

                    # ✅ Extract style: everything between codepoint and hash, excluding safe char if present
                    # We need to determine if safe char is present
                    safe_char_expected = (
                        char_part
                        if char_part.isprintable() and char_part not in '<>:"/\\|?*'
                        else ""
                    )

                    if len(parts) >= 4 and parts[1] == safe_char_expected:
                        # Format: U+XXXX_char_style_hash
                        # Style is parts[2:-1]
                        style_part = "_".join(parts[2:-1])
                    else:
                        # Format: U+XXXX_style_hash (no safe char)
                        # Style is parts[1:-1]
                        style_part = "_".join(parts[1:-1])

                    # Validate that extracted style matches folder name
                    if style_part != style_name:
                        tqdm.write(
                            f"  ⚠️  Style mismatch: extracted '{style_part}' != folder '{style_name}' in {filename}"
                        )
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

        print(f"\n✓ Initial scan:")
        print(f"  Styles: {len(styles_list)}")
        print(f"  Characters: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")

        # ✅ Auto-detect font parameter from existing content images
        if chars_list:
            self.detected_font = detect_font_parameter(content_dir, chars_list)
        else:
            print(f"  ⚠️  No characters found, cannot detect font parameter")
            self.detected_font = ""

        # Check for missing content images with detected font
        print(f"\n🔍 Validating content images (using font='{self.detected_font}')...")
        missing_content = []
        found_content = 0

        for char in tqdm(chars_list, desc="Checking content", unit="char", ncols=100):
            content_filename = get_content_filename(char, font=self.detected_font)
            content_path = content_dir / content_filename

            if not content_path.exists():
                missing_content.append(char)
            else:
                found_content += 1

        if missing_content:
            print(
                f"\n⚠️  WARNING: {len(missing_content)}/{len(chars_list)} characters missing content images:"
            )
            print(
                f"  Examples: {missing_content[:10]}{'...' if len(missing_content) > 10 else ''}"
            )
            print(f"  These characters will be excluded from splits")

            # Remove characters with missing content images
            for char in missing_content:
                del char_to_styles[char]
                characters.discard(char)
                valid_pairs = {(c, s) for c, s in valid_pairs if c != char}

            chars_list = sorted(list(characters))

        print(f"\n✅ After validation:")
        print(f"  Characters with content images: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")
        print(f"  Content images found: {found_content}")

        # Show character distribution per style
        style_char_counts = {
            style: len([c for c, s in valid_pairs if s == style])
            for style in styles_list
        }
        print(f"\n📊 Character distribution per style:")
        for style in sorted(style_char_counts.keys())[:10]:
            print(f"  {style}: {style_char_counts[style]} characters")
        if len(style_char_counts) > 10:
            print(f"  ... and {len(style_char_counts) - 10} more styles")

        return styles_list, chars_list, dict(char_to_styles), valid_pairs

    def create_simple_splits(
        self,
        styles: List[str],
        characters: List[str],
        char_to_styles: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        ✅ SIMPLIFIED: Create only train + val (unseen char + unseen style)
        """
        print("\n" + "=" * 60)
        print("CREATING TRAIN/VAL SPLITS")
        print("=" * 60)

        num_styles = len(styles)
        num_chars = len(characters)

        # Calculate validation split sizes
        num_val_styles = max(1, int(num_styles * self.config.val_split_ratio))
        num_val_chars = max(1, int(num_chars * self.config.val_split_ratio))
        
        num_train_styles = num_styles - num_val_styles
        num_train_chars = num_chars - num_val_chars

        # Randomly split
        shuffled_styles = styles.copy()
        random.shuffle(shuffled_styles)

        shuffled_chars = characters.copy()
        random.shuffle(shuffled_chars)

        train_styles = set(shuffled_styles[:num_train_styles])
        val_styles = set(shuffled_styles[num_train_styles:])

        train_chars = set(shuffled_chars[:num_train_chars])
        val_chars = set(shuffled_chars[num_train_chars:])

        scenarios = {
            "train": {
                "styles": list(train_styles),
                "characters": list(train_chars),
                "description": "Training data (seen styles + seen characters)",
            },
            "val": {
                "styles": list(val_styles),
                "characters": list(val_chars),
                "description": "Validation data (unseen styles + unseen characters)",
            },
        }

        print("\n📊 Split Statistics:")
        print(f"  Styles: {num_train_styles} train + {num_val_styles} val")
        print(f"  Chars:  {num_train_chars} train + {num_val_chars} val")

        print("\n📋 Splits:")
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n  {scenario_name}:")
            print(f"    Description: {scenario_data['description']}")
            print(f"    Styles: {len(scenario_data['styles'])} ({scenario_data['styles'][:3]}...)")
            print(f"    Chars: {len(scenario_data['characters'])} ({scenario_data['characters'][:5]}...)")

        return scenarios

    def copy_images_for_split(
        self,
        split_name: str,
        split_dir: Path,
        scenarios: Dict[str, Dict],
        valid_pairs: Set[Tuple[str, str]],
    ) -> Tuple[int, int, int]:
        """Copy images for a specific split with hash-based filename support"""
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

        # Identify valid pairs for this split
        split_valid_pairs = set()
        for char, style in valid_pairs:
            if char in allowed_chars and style in allowed_styles:
                split_valid_pairs.add((char, style))

        if not split_valid_pairs:
            print(f"  ⚠️  No valid pairs found for {split_name}")
            return 0, 0, 0

        print(f"  Valid pairs: {len(split_valid_pairs)}")

        chars_in_split = {char for char, style in split_valid_pairs}
        styles_in_split = {style for char, style in split_valid_pairs}

        print(f"  Unique chars: {len(chars_in_split)}")
        print(f"  Unique styles: {len(styles_in_split)}")

        # Copy content images
        source_content_dir = self.source_train_dir / "ContentImage"
        content_copied = 0
        content_missing = 0

        print(f"\n  📥 Copying content images (font='{self.detected_font}')...")
        for char in tqdm(
            sorted(chars_in_split), desc="  Content", ncols=80, unit="char", leave=False
        ):
            content_filename = get_content_filename(char, font=self.detected_font)
            src_path = source_content_dir / content_filename
            dst_path = split_content_dir / content_filename

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing: {src_path}")
                content_missing += 1
                continue

            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                content_copied += 1
            else:
                content_copied += 1

        if content_missing > 0:
            print(f"  ⚠️  Missing {content_missing} content images")

        # Copy target images
        source_target_dir = self.source_train_dir / "TargetImage"
        target_copied = 0
        skipped_pairs = 0

        print(f"  📥 Copying target images (font='{self.detected_font}')...")
        for char, style in tqdm(
            sorted(split_valid_pairs),
            desc="  Target",
            ncols=80,
            unit="pair",
            leave=False,
        ):
            style_dir = source_target_dir / style
            if not style_dir.exists():
                tqdm.write(f"    ⚠️  Style dir missing: {style_dir}")
                skipped_pairs += 1
                continue

            target_filename = get_target_filename(char, style, font=self.detected_font)
            src_path = style_dir / target_filename

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing: {src_path}")
                skipped_pairs += 1
                continue

            dst_path = split_target_dir / style / target_filename

            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                target_copied += 1
            else:
                target_copied += 1

        # Validate split
        print(f"\n  🔍 Validating split...")
        try:
            self._validate_split(split_dir)
        except ValueError as e:
            print(f"  ❌ Validation failed: {e}")

        # Summary
        print(f"\n  📊 Summary:")
        print(f"    Content copied: {content_copied}")
        print(f"    Target copied: {target_copied}")
        print(f"    Skipped: {skipped_pairs}")
        if content_missing > 0:
            print(f"    Missing content: {content_missing}")

        return content_copied, target_copied, skipped_pairs

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

                if "_" not in filename:
                    continue

                try:
                    parts = filename.split("_")
                    
                    # Extract codepoint (first part)
                    codepoint = parts[0]
                    
                    if not codepoint.startswith("U+"):
                        continue
                    
                    # Decode character from codepoint
                    char_code = int(codepoint.replace("U+", ""), 16)
                    char_part = chr(char_code)

                    # Check if content exists
                    content_filename = get_content_filename(char_part, font=self.detected_font)
                    content_path = content_dir / content_filename

                    if not content_path.exists():
                        missing_pairs += 1
                        tqdm.write(f"    ❌ Missing content: {content_filename} for {target_file.name}")
                        
                except (IndexError, ValueError) as e:
                    continue

        if missing_pairs > 0:
            print(f"  ⚠️  {missing_pairs}/{total_targets} targets missing content!")
            raise ValueError(f"Validation failed: {missing_pairs} missing content images")
        else:
            print(f"  ✓ All {total_targets} targets have matching content")

    def create_splits(self) -> None:
        """Create train and val splits"""
        print("\n" + "=" * 60)
        print("CREATING DATA SPLITS")
        print("=" * 60)

        # Analyze data and get valid pairs
        styles, characters, char_to_styles, valid_pairs = self.analyze_data()

        # Create simple train/val scenarios
        scenarios = self.create_simple_splits(styles, characters, char_to_styles)

        # Create train split
        print("\n📁 Creating train split:")
        train_content, train_target, train_skipped = self.copy_images_for_split(
            "train", self.train_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {train_content} content + {train_target} target (skipped {train_skipped})")

        # Create val split
        print(f"\n📁 Creating val split:")
        val_content, val_target, val_skipped = self.copy_images_for_split(
            "val", self.val_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {val_content} content + {val_target} target (skipped {val_skipped})")

        # Save metadata
        self._save_metadata(scenarios)

    def _save_metadata(self, scenarios: Dict[str, Dict]) -> None:
        """Save split information to JSON"""
        metadata_path = self.data_root / "split_info.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Saved split metadata to {metadata_path}")


def create_validation_split(
    data_root: str,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
) -> None:
    """
    Create validation splits (train + val only, unseen char + unseen style)
    ✅ Simplified version with no reliance on safe character in filenames
    """

    config = ValidationSplitConfig(
        data_root=data_root,
        val_split_ratio=val_split_ratio,
        random_seed=random_seed,
    )

    creator = ValidationSplitCreator(config)
    creator.create_splits()

    print("\n" + "=" * 60)
    print("✓ SPLIT CREATION COMPLETE")
    print("=" * 60)
    print("\n✅ Created directories:")
    print("  📁 train/ - Training data (seen styles + seen chars)")
    print("  📁 val/ - Validation data (unseen styles + unseen chars)")
    print("\n💡 Each folder guarantees:")
    print("  ✓ Every target image has matching content image")
    print("  ✓ Parsed by codepoint only (no reliance on safe char)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val splits with proper matching"
    )
    parser.add_argument(
        "--data_root", type=str, default="data_examples", help="Root data directory"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION SPLIT CREATOR (SIMPLIFIED)")
    print("=" * 60)

    try:
        create_validation_split(
            data_root=args.data_root,
            val_split_ratio=args.val_ratio,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

"""Example
python create_validation_split.py \
  --data_root /content/my_dataset \
  --val_ratio 0.2 \
  --seed 42
"""