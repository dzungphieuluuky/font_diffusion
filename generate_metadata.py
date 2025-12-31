"""
Generate results.json and results_checkpoint.json from existing dataset structure
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re


def parse_filename(filename: str) -> Tuple[int, str]:
    """
    Parse style and character index from filename
    Format: styleX+charY.png -> (Y, 'styleX')
    """
    match = re.match(r"style(\d+)\+char(\d+)\.png", filename)
    if match:
        style_idx = int(match.group(1))
        char_idx = int(match.group(2))
        return char_idx, f"style{style_idx}"
    return None, None


def generate_metadata_from_directory(
    data_root: str, output_dir: str = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Generate complete metadata from directory structure

    Args:
        data_root: Path to train_original directory
        output_dir: Output directory (defaults to data_root)

    Returns:
        Tuple of (results_data, checkpoint_data)
    """

    if output_dir is None:
        output_dir = data_root

    data_root = Path(data_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Reading from: {data_root}")

    # Initialize metadata
    generations: List[Dict[str, Any]] = []
    char_index_map: Dict[str, int] = {}
    style_index_map: Dict[str, int] = {}
    characters_set = set()
    styles_set = set()
    fonts_set = set()

    next_char_idx = 0

    # Get all fonts (directories in data_root)
    content_dir = data_root / "ContentImage"
    target_dir = data_root / "TargetImage"

    if not content_dir.exists() or not target_dir.exists():
        print(f"❌ Missing required directories:")
        print(f"   {content_dir} exists: {content_dir.exists()}")
        print(f"   {target_dir} exists: {target_dir.exists()}")
        return {}, {}

    # Parse content images to build char mapping
    print(f"\n📝 Parsing content images...")
    content_files = list(content_dir.glob("char*.png"))
    print(f"   Found {len(content_files)} content images")

    for content_file in sorted(content_files):
        # Extract char index from filename (charX.png)
        match = re.match(r"char(\d+)\.png", content_file.name)
        if match:
            char_idx = int(match.group(1))
            # Use the actual character or generate placeholder
            char = f"char{char_idx}"
            char_index_map[char] = char_idx
            characters_set.add(char)

    print(f"   Mapped {len(char_index_map)} characters")

    # Parse target images to build complete generation records
    print(f"\n🎨 Parsing target images...")
    target_subdirs = [d for d in target_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(target_subdirs)} style directories")

    for style_dir in sorted(target_subdirs):
        style_name = style_dir.name  # e.g., "style0"

        # Extract style index
        match = re.match(r"style(\d+)", style_name)
        if match:
            style_idx = int(match.group(1))
            style_index_map[style_name] = style_idx
            styles_set.add(style_name)

            # Get all target images for this style
            target_files = list(style_dir.glob("*.png"))
            print(f"   {style_name}: {len(target_files)} images")

            for target_file in sorted(target_files):
                char_idx, parsed_style = parse_filename(target_file.name)

                if char_idx is not None:
                    char = f"char{char_idx}"

                    generation = {
                        "character": char,
                        "char_index": char_idx,
                        "style": style_name,
                        "style_index": style_idx,
                        "font": "NomNaTongLight2",  # Update this based on your font
                        "output_path": str(target_file.relative_to(data_root)),
                        "content_image_path": f"ContentImage/char{char_idx}.png",
                        "target_image_path": f"TargetImage/{style_name}/{style_name}+char{char_idx}.png",
                    }
                    generations.append(generation)

    print(f"\n✅ Generated {len(generations)} generation records")

    # Build final metadata
    results_metadata = {
        "generations": generations,
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "characters": sorted(list(characters_set)),
        "styles": sorted(list(styles_set)),
        "fonts": ["NomNaTongLight2"],
        "total_chars": len(characters_set),
        "total_styles": len(styles_set),
    }

    # Checkpoint is identical to results
    checkpoint_metadata = results_metadata.copy()

    return results_metadata, checkpoint_metadata


def save_metadata_files(
    results_data: Dict[str, Any], checkpoint_data: Dict[str, Any], output_dir: str
) -> None:
    """Save metadata files"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results.json
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Saved: {results_path}")
    print(f"   Generations: {len(results_data.get('generations', []))}")

    # Save results_checkpoint.json
    checkpoint_path = output_dir / "results_checkpoint.json"
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved: {checkpoint_path}")
    print(f"   Generations: {len(checkpoint_data.get('generations', []))}")


if __name__ == "__main__":
    # Configure paths
    data_root = "my_dataset/train_original"

    print("=" * 60)
    print("GENERATING METADATA FROM DATASET")
    print("=" * 60)

    # Generate metadata
    results_data, checkpoint_data = generate_metadata_from_directory(data_root)

    # Save files
    if results_data:
        save_metadata_files(results_data, checkpoint_data, data_root)

        print("\n" + "=" * 60)
        print("✓ COMPLETE!")
        print("=" * 60)
        print(f"\nMetadata Summary:")
        print(f"  Total characters: {results_data.get('total_chars', 0)}")
        print(f"  Total styles: {results_data.get('total_styles', 0)}")
        print(f"  Total generations: {len(results_data.get('generations', []))}")
    else:
        print("\n❌ Failed to generate metadata")
