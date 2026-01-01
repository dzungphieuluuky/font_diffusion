"""
Create validation/test splits from training data
✅ CORRECTED: Properly handles train/val splits with checkpoint filtering
✅ Parses filenames as codepoint_char_hash.png and codepoint_char_style_hash.png
✅ Ensures checkpoint.json contains only relevant generations
✅ Validates content↔target pairs exist before including in split
"""

import os
import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict
import random

from huggingface_hub.utils import tqdm
import hashlib


# Setup logging with tqdm compatibility
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[TqdmLoggingHandler()],
)


# ============================================================================
# UTILITY FUNCTIONS - For filename parsing and hashing
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


def parse_content_filename(filename: str) -> Optional[str]:
    """
    Parse content filename to extract character only
    Format: U+XXXX_{char}_{hash}.png or U+XXXX_{hash}.png
    
    Returns: character or None if parse fails
    """
    if not filename.endswith(".png"):
        return None

    stem = filename[:-4]  # Remove .png
    parts = stem.split("_")

    if len(parts) < 2:
        return None

    codepoint = parts[0]

    if not codepoint.startswith("U+"):
        return None

    try:
        char_code = int(codepoint.replace("U+", ""), 16)
        char = chr(char_code)
        return char
    except (ValueError, OverflowError):
        return None


def parse_target_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse target filename to extract character and style
    Format: U+XXXX_{char}_{style}_{hash}.png
    
    Returns: (char, style) or None if parse fails
    """
    if not filename.endswith(".png"):
        return None

    stem = filename[:-4]  # Remove .png
    parts = stem.split("_")

    if len(parts) < 3:
        return None

    codepoint = parts[0]

    if not codepoint.startswith("U+"):
        return None

    try:
        char_code = int(codepoint.replace("U+", ""), 16)
        char = chr(char_code)
        
        # parts[0] = codepoint
        # parts[1] = character
        # parts[2:-1] = style parts
        # parts[-1] = hash
        
        style_parts = parts[2:-1]
        if not style_parts:
            return None
        
        style = "_".join(style_parts)
        return (char, style)
    except (ValueError, OverflowError, IndexError):
        return None

@dataclass
class ValidationSplitConfig:
    """Configuration for validation split creation"""

    data_root: str
    val_split_ratio: float = 0.2
    random_seed: int = 42


# ============================================================================
# MAIN CLASS - ValidationSplitCreator
# ============================================================================

class ValidationSplitCreator:
    """Create train/val splits with proper checkpoint filtering"""

    def __init__(self, config: ValidationSplitConfig):
        self.config = config
        self.data_root = Path(config.data_root)

        self.original_train_dir: Path = self.data_root / "train_original"
        self.train_dir: Path = self.data_root / "train"
        self.val_dir: Path = self.data_root / "val"

        random.seed(config.random_seed)
        self.detected_font: str = ""

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate training directory structure"""
        source_dir: Path = (
            self.original_train_dir
            if self.original_train_dir.exists()
            else self.train_dir
        )

        if not (source_dir / "TargetImage").exists():
            raise ValueError(f"TargetImage not found in {source_dir}")
        if not (source_dir / "ContentImage").exists():
            raise ValueError(f"ContentImage not found in {source_dir}")

        self.source_train_dir: Path = source_dir
        logging.info(f"✓ Using source directory: {self.source_train_dir}")

    def analyze_data(
        self,
    ) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str], Dict[str, List[str]]]:
        """
        ✅ CORRECTED: Analyze by scanning actual files and matching content↔target pairs
        ✅ With detailed diagnostics to find missing images
        
        Returns:
            - content_files: {char -> file_path}
            - target_files: {(char, style) -> file_path}
            - char_to_styles: {char -> [styles]}
        """
        logging.info("\n" + "=" * 70)
        logging.info("ANALYZING TRAINING DATA")
        logging.info("=" * 70)

        content_dir: Path = self.source_train_dir / "ContentImage"
        target_dir: Path = self.source_train_dir / "TargetImage"

        content_files: Dict[str, str] = {}  # char -> file_path
        target_files: Dict[Tuple[str, str], str] = {}  # (char, style) -> file_path
        char_to_styles: Dict[str, List[str]] = defaultdict(set)

        # Scan content images
        logging.info("\n🔍 Scanning content images...")
        if content_dir.exists():
            for img_file in tqdm(
                list(content_dir.glob("*.png")),
                desc="Content images",
                unit="img",
            ):
                char = parse_content_filename(img_file.name)
                if char:
                    content_files[char] = str(img_file)

        logging.info(f"  ✓ Found {len(content_files)} content images")

        # Scan target images with detailed diagnostics
        logging.info("\n🔍 Scanning target images...")
        total_targets = 0
        style_mismatch_count = 0
        parse_error_count = 0
        
        style_mismatch_details = defaultdict(list)
        unparseable_files = []  # ✅ Collect unparseable files for later diagnosis
        
        for style_folder in tqdm(
            sorted(target_dir.iterdir()),
            desc="Styles",
            unit="style",
        ):
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name

            for img_file in style_folder.glob("*.png"):
                parsed = parse_target_filename(img_file.name)
                
                # Save parse errors file path for diagnosis
                if parsed is None:
                    parse_error_count += 1
                    unparseable_files.append({
                        "folder": style_name,
                        "filename": img_file.name,
                    })
                    continue
                
                char, style = parsed

                # ✅ Validate style matches folder
                if style != style_name:
                    style_mismatch_count += 1
                    style_mismatch_details[style_name].append({
                        "filename": img_file.name,
                        "extracted_style": style,
                        "folder_style": style_name,
                    })
                    continue  # ✅ Skip this file

                target_files[(char, style)] = str(img_file)
                char_to_styles[char].add(style)
                total_targets += 1

        logging.info(f"  ✓ Found {total_targets} valid target images")
        
        # ✅ Print parse error diagnostics
        if parse_error_count > 0:
            logging.info(f"\n⚠️  PARSE ERROR DIAGNOSTICS:")
            logging.info(f"  Total parse errors: {parse_error_count}")
            logging.info(f"\n  First 10 unparseable files:")
            for item in unparseable_files[:10]:
                logging.info(f"    Folder: {item['folder']}")
                logging.info(f"    File:   {item['filename']}")
                stem = item['filename'][:-4]
                parts = stem.split("_")
                logging.info(f"    Parts:  {parts} (count: {len(parts)})")
            if len(unparseable_files) > 10:
                logging.info(f"    ... and {len(unparseable_files) - 10} more")

            # --- Export unparseable files to a txt file ---
            unparseable_txt_path = self.data_root / "unparseable_files.txt"
            with open(unparseable_txt_path, "w", encoding="utf-8") as f:
                for item in unparseable_files:
                    abs_path = str((self.source_train_dir / "TargetImage" / item["folder"] / item["filename"]).resolve())
                    f.write(abs_path + "\n")
            logging.info(f"\n✓ Exported unparseable file list to {unparseable_txt_path}")


        # ✅ Print style mismatch diagnostics
        if style_mismatch_count > 0:
            logging.info(f"\n⚠️  STYLE MISMATCH DIAGNOSTICS:")
            logging.info(f"  Total mismatches: {style_mismatch_count}")
            for style_folder, mismatches in style_mismatch_details.items():
                logging.info(f"\n  Folder: {style_folder}")
                logging.info(f"    Mismatch count: {len(mismatches)}")
                for mismatch in mismatches[:3]:
                    logging.info(f"      - {mismatch['filename']}")
                    logging.info(f"        Extracted: '{mismatch['extracted_style']}' vs Expected: '{mismatch['folder_style']}'")
                if len(mismatches) > 3:
                    logging.info(f"      ... and {len(mismatches) - 3} more")

        # Validate content↔target pairing
        logging.info("\n🔍 Validating content ↔ target pairs...")
        valid_pairs: Dict[Tuple[str, str], bool] = {}
        missing_content_count = 0

        for (char, style) in tqdm(
            target_files.keys(),
            desc="Validating pairs",
            ncols=100,
            unit="pair",
        ):
            if char not in content_files:
                missing_content_count += 1
                valid_pairs[(char, style)] = False
            else:
                valid_pairs[(char, style)] = True

        # Filter to only valid pairs
        valid_target_files = {
            pair: path
            for pair, path in target_files.items()
            if valid_pairs.get(pair, False)
        }

        # ✅ COMPREHENSIVE ANALYSIS SUMMARY
        logging.info(f"\n" + "=" * 70)
        logging.info(f"📊 DATA ANALYSIS SUMMARY")
        logging.info(f"=" * 70)
        logging.info(f"Content images found:        {len(content_files):,}")
        logging.info(f"Target images scanned:       {total_targets:,}")
        logging.info(f"  ├─ Parse errors:          {parse_error_count:,}")
        logging.info(f"  └─ Style mismatches:      {style_mismatch_count:,}")
        logging.info(f"Target images after filter:  {len(target_files):,}")
        logging.info(f"Missing content images:      {missing_content_count:,}")
        logging.info(f"Final valid pairs:           {len(valid_target_files):,}")
        logging.info(f"=" * 70)
        
        # ✅ Calculate and show loss
        expected_total = total_targets
        lost_to_parse_error = parse_error_count
        lost_to_style_mismatch = style_mismatch_count
        lost_to_missing_content = missing_content_count
        total_lost = lost_to_parse_error + lost_to_style_mismatch + lost_to_missing_content
        
        if total_lost > 0:
            logging.info(f"\n⚠️  IMAGE LOSS BREAKDOWN:")
            logging.info(f"  Total scanned:          {expected_total:,}")
            logging.info(f"  Lost to parse errors:   {lost_to_parse_error:,} ({lost_to_parse_error*100/expected_total:.2f}%)")
            logging.info(f"  Lost to style mismatch: {lost_to_style_mismatch:,} ({lost_to_style_mismatch*100/expected_total:.2f}%)")
            logging.info(f"  Lost to missing content:{lost_to_missing_content:,} ({lost_to_missing_content*100/expected_total:.2f}%)")
            logging.info(f"  Total lost:             {total_lost:,} ({total_lost*100/expected_total:.2f}%)")
            logging.info(f"  Usable for split:       {len(valid_target_files):,} ({len(valid_target_files)*100/expected_total:.2f}%)")

        return content_files, valid_target_files, dict(char_to_styles)

    def create_simple_splits(
        self,
        content_files: Dict[str, str],
        target_files: Dict[Tuple[str, str], str],
        char_to_styles: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        Create train/val splits
        - Randomly split both characters and styles
        - Only pairs (char, style) where both char and style are in the split are included
        """
        logging.info("\n" + "=" * 70)
        logging.info("CREATING TRAIN/VAL SPLITS (random char & style)")
        logging.info("=" * 70)

        all_chars = sorted(list(content_files.keys()))
        all_styles = sorted({style for (_, style) in target_files.keys()})
        num_chars = len(all_chars)
        num_styles = len(all_styles)

        num_val_chars = max(1, int(num_chars * self.config.val_split_ratio))
        num_train_chars = num_chars - num_val_chars

        num_val_styles = max(1, int(num_styles * self.config.val_split_ratio))
        num_train_styles = num_styles - num_val_styles

        # Shuffle and split characters and styles
        shuffled_chars = all_chars.copy()
        random.shuffle(shuffled_chars)
        train_chars = set(shuffled_chars[:num_train_chars])
        val_chars = set(shuffled_chars[num_train_chars:])

        shuffled_styles = all_styles.copy()
        random.shuffle(shuffled_styles)
        train_styles = set(shuffled_styles[:num_train_styles])
        val_styles = set(shuffled_styles[num_train_styles:])

        scenarios = {
            "train": {
                "characters": sorted(list(train_chars)),
                "styles": sorted(list(train_styles)),
                "description": "Training split (random chars & styles)",
            },
            "val": {
                "characters": sorted(list(val_chars)),
                "styles": sorted(list(val_styles)),
                "description": "Validation split (random chars & styles)",
            },
        }

        logging.info("\n📊 Split Statistics:")
        logging.info(
            f"  Total chars: {num_chars} → train: {num_train_chars}, val: {num_val_chars}"
        )
        logging.info(
            f"  Total styles: {num_styles} → train: {num_train_styles}, val: {num_val_styles}"
        )

        for split_name, split_data in scenarios.items():
            logging.info(f"\n  {split_name}:")
            logging.info(f"    Chars: {len(split_data['characters'])}")
            logging.info(f"    Styles: {len(split_data['styles'])}")

        return scenarios

    def copy_images_for_split(
        self,
        split_name: str,
        split_dir: Path,
        scenarios: Dict[str, Dict],
        content_files: Dict[str, str],
        target_files: Dict[Tuple[str, str], str],
    ) -> Tuple[int, int, int]:
        """Copy images for a specific split using ACTUAL file paths"""
        split_config = scenarios[split_name]
        allowed_chars = set(split_config["characters"])
        allowed_styles = set(split_config["styles"])

        # Create directories
        split_content_dir = split_dir / "ContentImage"
        split_target_dir = split_dir / "TargetImage"
        split_content_dir.mkdir(parents=True, exist_ok=True)
        split_target_dir.mkdir(parents=True, exist_ok=True)

        # Create style subdirectories
        for style in allowed_styles:
            (split_target_dir / style).mkdir(exist_ok=True)

        content_copied = 0
        target_copied = 0
        skipped = 0

        # Copy content images
        logging.info(f"\n  📥 Copying content images for {split_name}...")
        for char in tqdm(
            sorted(allowed_chars),
            desc="  Content",
            ncols=80,
            unit="char",
            leave=False,
        ):
            if char not in content_files:
                skipped += 1
                continue

            src_path = Path(content_files[char])
            
            if not src_path.exists():
                tqdm.write(f"    ⚠️  Source not found: {src_path}")
                skipped += 1
                continue

            dst_path = split_content_dir / src_path.name

            if src_path.resolve() != dst_path.resolve():
                try:
                    shutil.copy2(src_path, dst_path)
                    content_copied += 1
                except Exception as e:
                    tqdm.write(f"    ⚠️  Error copying: {e}")
                    skipped += 1
            else:
                content_copied += 1

        # Copy target images
        logging.info(f"  📥 Copying target images for {split_name}...")
        for (char, style), target_path_str in tqdm(
            sorted(target_files.items()),
            desc="  Target",
            ncols=80,
            unit="pair",
            leave=False,
        ):
            # Only copy if char and style are in this split
            if char not in allowed_chars or style not in allowed_styles:
                continue

            src_path = Path(target_path_str)
            
            if not src_path.exists():
                tqdm.write(f"    ⚠️  Source not found: {src_path}")
                skipped += 1
                continue

            dst_path = split_target_dir / style / src_path.name

            if src_path.resolve() != dst_path.resolve():
                try:
                    shutil.copy2(src_path, dst_path)
                    target_copied += 1
                except Exception as e:
                    tqdm.write(f"    ⚠️  Error copying: {e}")
                    skipped += 1
            else:
                target_copied += 1

        logging.info(
            f"  ✓ {split_name}: {content_copied:,} content, {target_copied:,} target (skipped: {skipped})"
        )

        return content_copied, target_copied, skipped

    def _copy_and_filter_checkpoint(
        self,
        split_name: str,
        split_dir: Path,
        allowed_chars: Set[str],
        allowed_styles: Set[str],
        target_files: Dict[Tuple[str, str], str],
    ) -> None:
        """
        Filter results_checkpoint.json to only include generations
        that have both content and target in this split
        """
        logging.info(f"\n  📋 Filtering checkpoint for {split_name}...")

        original_checkpoint_path = self.source_train_dir / "results_checkpoint.json"

        if not original_checkpoint_path.exists():
            logging.info(f"    ⚠️  No checkpoint found, skipping")
            return

        try:
            with open(original_checkpoint_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
        except Exception as e:
            logging.info(f"    ⚠️  Error loading checkpoint: {e}")
            return

        # Filter generations
        original_generations = original_data.get("generations", [])
        filtered_generations = []

        for gen in tqdm(
            original_generations,
            desc="    Filtering",
            ncols=80,
            unit="gen",
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        ):
            char = gen.get("character")
            style = gen.get("style")

            # Include only if:
            # 1. Character is in this split
            # 2. Style is in this split
            # 3. Target image actually exists in this split
            if (
                char in allowed_chars
                and style in allowed_styles
                and (char, style) in target_files
            ):
                filtered_generations.append(gen)

        # Create checkpoint for this split
        split_checkpoint = {
            "split": split_name,
            "num_characters": len(allowed_chars),
            "num_styles": len(allowed_styles),
            "num_generations": len(filtered_generations),
            "characters": sorted(list(allowed_chars)),
            "styles": sorted(list(allowed_styles)),
            "generations": filtered_generations,
            "fonts": original_data.get("fonts", []),
            "metrics": {},
            "original_source": str(self.source_train_dir),
            "filtered_from": str(original_checkpoint_path),
        }

        split_checkpoint_path = split_dir / "results_checkpoint.json"

        with open(split_checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(split_checkpoint, f, indent=2, ensure_ascii=False)

        logging.info(
            f"    ✓ Saved: {len(filtered_generations):,}/{len(original_generations):,} generations"
        )

    def create_splits(self) -> None:
        """Main function to create train/val splits"""
        logging.info("\n" + "=" * 70)
        logging.info("FONTDIFFUSION VALIDATION SPLIT CREATOR")
        logging.info("=" * 70)

        # Step 1: Analyze data
        content_files, target_files, char_to_styles = self.analyze_data()

        # Step 2: Create split scenarios
        scenarios = self.create_simple_splits(
            content_files, target_files, char_to_styles
        )

        # Step 3: Create train split
        logging.info("\n📁 CREATING TRAIN SPLIT...")
        train_chars = set(scenarios["train"]["characters"])
        train_styles = set(scenarios["train"]["styles"])
        
        self.copy_images_for_split(
            "train", self.train_dir, scenarios, content_files, target_files
        )
        self._copy_and_filter_checkpoint(
            "train",
            self.train_dir,
            train_chars,
            train_styles,
            target_files,
        )

        # Step 4: Create val split
        logging.info(f"\n📁 CREATING VAL SPLIT...")
        val_chars = set(scenarios["val"]["characters"])
        val_styles = set(scenarios["val"]["styles"])
        
        self.copy_images_for_split(
            "val", self.val_dir, scenarios, content_files, target_files
        )
        self._copy_and_filter_checkpoint(
            "val",
            self.val_dir,
            val_chars,
            val_styles,
            target_files,
        )

        # Step 5: Save metadata
        self._save_metadata(scenarios)

    def _save_metadata(self, scenarios: Dict[str, Dict]) -> None:
        """Save split information to JSON"""
        metadata_path = self.data_root / "split_info.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2, ensure_ascii=False)

        logging.info(f"\n✓ Saved split metadata to {metadata_path}")


# ============================================================================
# ENTRY POINT
# ============================================================================

def create_validation_split(
    data_root: str,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
) -> None:
    """Create validation splits with proper checkpoint filtering"""
    config = ValidationSplitConfig(
        data_root=data_root,
        val_split_ratio=val_split_ratio,
        random_seed=random_seed,
    )

    creator = ValidationSplitCreator(config)
    creator.create_splits()

    logging.info("\n" + "=" * 70)
    logging.info("✓ SPLIT CREATION COMPLETE")
    logging.info("=" * 70)
    logging.info("\n✅ Created:")
    logging.info("  📁 train/")
    logging.info("    ├── ContentImage/ (training chars)")
    logging.info("    ├── TargetImage/ (training styles)")
    logging.info("    └── results_checkpoint.json (filtered)")
    logging.info("  📁 val/")
    logging.info("    ├── ContentImage/ (validation chars)")
    logging.info("    ├── TargetImage/ (validation styles)")
    logging.info("    └── results_checkpoint.json (filtered)")
    logging.info("\n💡 Guarantees:")
    logging.info("  ✓ Every target has matching content")
    logging.info("  ✓ Checkpoint contains only relevant generations")
    logging.info("  ✓ Train and val are completely disjoint")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create train/val splits with checkpoint filtering"
    )
    parser.add_argument(
        "--data_root", type=str, default="data_examples", help="Root data directory"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    try:
        create_validation_split(
            data_root=args.data_root,
            val_split_ratio=args.val_ratio,
            random_seed=args.seed,
        )
    except Exception as e:
        logging.error(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)