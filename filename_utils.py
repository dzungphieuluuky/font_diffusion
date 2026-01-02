"""
✅ UNIFIED FILENAME UTILITIES
Centralized functions for the new simplified naming convention:
  - ContentImage: {char}.png
  - TargetImage: {style}+{char}.png
"""

import os
from pathlib import Path
import hashlib
from typing import Optional, Tuple

def compute_file_hash(char: str, style: str, font: str = "") -> str:
    """Compute deterministic hash for a (character, style, font) combination"""
    content = f"{char}_{style}_{font}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

def get_content_filename(char: str) -> str:
    """
    Get content image filename for character
    Format: {char}.png
    Example: 中.png
    No hashing, no codepoints, just the character
    """
    return f"{char}.png"


def get_target_filename(char: str, style: str) -> str:
    """
    Get target image filename
    Format: {style}+{char}.png
    Example: style0+中.png
    Just style+char, no hashing
    """
    return f"{style}+{char}.png"


def parse_content_filename(filename: str) -> Optional[str]:
    """Parse content filename to extract character
    Format: {char}.png

    Args:
        filename (str): Filename to parse

    Returns:
        Optional[str]: Character or None if parse fails
    """
    if not filename.endswith(".png"):
        return None
    
    # Remove .png extension
    char = filename[:-4]
    
    # Must be exactly one character
    if len(char) == 1:
        return char
    
    return None


def parse_target_filename(filename: str) -> Optional[Tuple[str, str]]:
    """
    Parse target filename to extract character and style
    Format: {style}+{char}.png
    Args:
        filename (str): Filename to parse
    Returns: 
        (char, style) tuple or None if parse fails
    Example:
        "style0+中.png" -> ("中", "style0")
    """
    if not filename.endswith(".png"):
        return None
    
    # Remove .png extension
    stem = filename[:-4]
    
    # Must contain exactly one '+'
    if stem.count("+") != 1:
        return None
    
    style, char = stem.split("+")
    
    # Character must be exactly 1 char, style must be non-empty
    if len(char) == 1 and len(style) > 0:
        return (char, style)
    
    return None


def content_file_exists(content_dir: Path, char: str) -> bool:
    """
    Check if content image exists for character
    
    Args:
        content_dir: Path to ContentImage directory
        char: Unicode character
    
    Returns:
        True if file exists
    """
    filename = get_content_filename(char)
    filepath = content_dir / filename
    return filepath.exists()


def target_file_exists(target_dir: Path, style: str, char: str) -> bool:
    """
    Check if target image exists for (char, style) pair
    
    Args:
        target_dir: Path to TargetImage directory
        style: Style name
        char: Unicode character
    
    Returns:
        True if file exists
    """
    filename = get_target_filename(char, style)
    filepath = target_dir / style / filename
    return filepath.exists()


def get_content_path(content_dir: Path, char: str) -> Path:
    """Get full path to content image"""
    return content_dir / get_content_filename(char)


def get_target_path(target_dir: Path, style: str, char: str) -> Path:
    """Get full path to target image"""
    return target_dir / style / get_target_filename(char, style)